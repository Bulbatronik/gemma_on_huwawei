/*
 * GemmaNano on-device inference runtime.
 *
 * Target: Huawei Lite Wearable JS runtime (JerryScript, ES5.1 + typed arrays,
 * no regex, no eval, ~48 KB JS heap).  Design rules:
 *   - weights never live in the heap: they are streamed block-by-block from
 *     internal://app/ files (the engine only says WHICH block it needs next and
 *     the caller feeds it, so file I/O can be async and the stack unwinds
 *     between blocks);
 *   - every heavy loop is int8 x int8 -> integer accumulate, which JerryScript
 *     keeps as unboxed "direct integers" (no heap allocation per op);
 *   - floats are only used for O(d) work (norms, RoPE, softmax, GELU).
 *
 * The same file runs under Node and under a desktop JerryScript build for tests
 * (tests strip the final "export default" line).
 *
 * Block layout (must match tools/gemma_nano/export.py):
 *   a row record = [fp16 scale (2 bytes, little-endian)][n int8 values];
 *   rows are packed in order into B-byte blocks and never straddle a block
 *   (if a row does not fit, the rest of the block is padding).
 *   Region E: embedding rows (V x d) starting at block 0.  Also the tied LM head.
 *   Region L: starting at block cfg.layerStart, per layer:
 *     norm(d) pre-attn, Q (Hq*hd rows x d), K (hd x d), V (hd x d),
 *     norm(hd) q-norm, norm(hd) k-norm, O (d x Hq*hd), norm(d) post-attn,
 *     norm(d) pre-ffn, gate/up interleaved (2*ffn rows x d), down (d x ffn),
 *     norm(d) post-ffn;  then norm(d) final.
 *   Norm rows store (1 + w) like Gemma's RMSNorm.
 */
var GN = (function () {
  var P2 = [];
  for (var e = 0; e < 32; e++) P2.push(Math.pow(2, e - 25));

  function f16(lo, hi) {
    var h = lo | (hi << 8);
    var ex = (h >> 10) & 31;
    var v = ex === 0 ? (h & 1023) * 5.960464477539063e-8 : (1024 + (h & 1023)) * P2[ex];
    return (h & 0x8000) ? -v : v;
  }

  // op types
  var T_NORM = 0, T_MM = 1, T_FN = 2, T_HEAD = 3;
  // norm targets
  var N_PRE_ATT = 0, N_Q = 1, N_K = 2, N_POST_ATT = 3, N_PRE_FFN = 4, N_POST_FFN = 5, N_FINAL = 6;
  // matmul outputs
  var M_Q = 0, M_K = 1, M_V = 2, M_O = 3, M_GU = 4, M_DOWN = 5;
  // functions
  var F_ATT = 0, F_QM = 1;

  function Engine(c) {
    var d = c.d, qd = c.Hq * c.hd, mx = Math.max(d, qd, c.ffn);
    this.c = c;
    this.B = c.B;
    this.need = [0, 0, 0, 0, 0, 0]; // row bytes per matmul code
    this.need[M_Q] = 2 + d; this.need[M_K] = 2 + d; this.need[M_V] = 2 + d;
    this.need[M_O] = 2 + qd; this.need[M_GU] = 2 + d; this.need[M_DOWN] = 2 + c.ffn;
    this.rpbE = Math.floor(c.B / (2 + d));
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.o = new Float32Array(d);
    this.q = new Float32Array(qd);
    this.k = new Float32Array(c.hd);
    this.v = new Float32Array(c.hd);
    this.att = new Float32Array(qd);
    this.m = new Float32Array(c.ffn);
    this.nw = new Float32Array(Math.max(d, c.hd));
    this.xq = new Int8Array(mx);
    this.qq = new Int8Array(c.hd);
    this.kc = new Int8Array(c.L * c.ctx * c.hd);
    this.vc = new Int8Array(c.L * c.ctx * c.hd);
    this.ks = new Float32Array(c.L * c.ctx);
    this.vs = new Float32Array(c.L * c.ctx);
    this.sc = new Float32Array(c.ctx);
    this.wi = new Int16Array(c.ctx);
    this.cs = new Float32Array(c.hd >> 1);
    this.sn = new Float32Array(c.hd >> 1);
    this.inv = new Float32Array(c.hd >> 1);
    for (var i = 0; i < (c.hd >> 1); i++) this.inv[i] = Math.pow(c.theta, -2 * i / c.hd);
    this.K = c.topk;
    this.tkI = new Int16Array(c.topk);
    this.tkV = new Float32Array(c.topk);
    this.rep = new Uint8Array(c.V);
    // program
    var ops = [];
    for (var l = 0; l < c.L; l++) {
      ops.push(T_NORM, d, N_PRE_ATT, T_MM, qd, M_Q, T_MM, c.hd, M_K, T_MM, c.hd, M_V,
        T_NORM, c.hd, N_Q, T_NORM, c.hd, N_K, T_FN, 0, F_ATT, T_MM, d, M_O,
        T_NORM, d, N_POST_ATT, T_NORM, d, N_PRE_FFN, T_MM, 2 * c.ffn, M_GU,
        T_FN, 0, F_QM, T_MM, d, M_DOWN, T_NORM, d, N_POST_FFN);
    }
    this.nLayerOps = ops.length;
    ops.push(T_NORM, d, N_FINAL, T_HEAD, c.V, 0);
    this.ops = new Int16Array(ops);
    this.done = true;
  }

  var P = Engine.prototype;

  /** Start processing token `tok` at position `pos`.  If wantLogits, the
   *  token's output distribution is computed and a next token is sampled. */
  P.start = function (tok, pos, wantLogits) {
    var c = this.c;
    if (pos >= c.ctx) throw new Error('ctx full');
    this.tok = tok; this.pos = pos;
    this.end = wantLogits ? this.ops.length : this.nLayerOps;
    this.pc = 0; this.ri = 0; this.lay = 0;
    this.emb = true;
    this.blk = Math.floor(tok / this.rpbE);
    this.off = (tok % this.rpbE) * (2 + c.d);
    this.cur = null; this.cur8 = null;
    this.nk = 0; this.out = -1;
    for (var i = 0; i < (c.hd >> 1); i++) {
      var a = pos * this.inv[i];
      this.cs[i] = Math.cos(a); this.sn[i] = Math.sin(a);
    }
    this.done = false;
  };

  /** Index of the block the engine needs next, or -1 when the token is done. */
  P.next = function () {
    return this.done ? -1 : this.blk;
  };

  /** Feed the bytes (Uint8Array) of block next(). */
  P.feed = function (u8) {
    this.cur = u8;
    this.cur8 = new Int8Array(u8.buffer, u8.byteOffset, u8.length);
    if (this.emb) {
      this.loadEmb();
      this.emb = false;
      this.blk = this.c.layerStart; this.off = 0; this.cur = null; this.cur8 = null;
      return;
    }
    this.run();
  };

  P.loadEmb = function () {
    var c = this.c, u = this.cur, w = this.cur8, o = this.off, x = this.x;
    var s = f16(u[o], u[o + 1]) * Math.sqrt(c.d);
    o += 2;
    for (var i = 0; i < c.d; i++) x[i] = w[o + i] * s;
  };

  function quant(src, n, dst) {
    var mx = 0, i, a;
    for (i = 0; i < n; i++) { a = src[i]; if (a < 0) a = -a; if (a > mx) mx = a; }
    if (mx === 0) { for (i = 0; i < n; i++) dst[i] = 0; return 0; }
    var s = 127 / mx;
    for (i = 0; i < n; i++) dst[i] = Math.round(src[i] * s);
    return mx / 127;
  }

  function rms(src, so, n, w, dst, eps) {
    var ss = 0, i, a;
    for (i = 0; i < n; i++) { a = src[so + i]; ss += a * a; }
    var r = 1 / Math.sqrt(ss / n + eps);
    for (i = 0; i < n; i++) dst[so + i] = src[so + i] * r * w[i];
  }

  function gelu(a) {
    var z = 1.5957691216057308 * (a + 0.044715 * a * a * a);
    // 0.5*a*(1+tanh(z/2)) == a * sigmoid(z)... written via exp, guarded for overflow
    if (z > 30) return a;
    if (z < -30) return 0;
    return a / (1 + Math.exp(-z));
  }

  P.run = function () {
    var ops = this.ops, c = this.c, B = this.B;
    while (true) {
      if (this.pc >= this.end) { this.finish(); return; }
      var pc = this.pc, t = ops[pc], a = ops[pc + 1], b = ops[pc + 2];
      if (t === T_FN) {
        if (b === F_ATT) this.attention(); else this.xs = quant(this.m, c.ffn, this.xq);
        this.pc += 3;
        continue;
      }
      if (this.cur === null) return;
      var u = this.cur, w = this.cur8, off = this.off, ri = this.ri, rows, need, r, k, acc, o;
      if (t === T_NORM) {
        need = 2 + a;
        if (off + need > B) { this.blk++; this.off = 0; this.cur = null; return; }
        var s = f16(u[off], u[off + 1]), nw = this.nw;
        o = off + 2;
        for (k = 0; k < a; k++) nw[k] = w[o + k] * s;
        this.off = off + need;
        this.applyNorm(b);
        this.pc += 3;
        continue;
      }
      if (t === T_MM) {
        rows = a; need = this.need[b];
        var cols = need - 2, x = this.xq, sIn = this.xs, out;
        if (b === M_Q) out = this.q; else if (b === M_K) out = this.k; else if (b === M_V) out = this.v;
        else if (b === M_O) out = this.o; else if (b === M_DOWN) out = this.o; else out = this.m;
        while (ri < rows) {
          if (off + need > B) { this.blk++; this.off = 0; this.ri = ri; this.cur = null; return; }
          acc = 0; o = off + 2;
          for (k = 0; k < cols; k++) acc += w[o + k] * x[k];
          var val = acc * f16(u[off], u[off + 1]) * sIn;
          if (b === M_GU) {
            if (ri & 1) out[ri >> 1] = gelu(this.g) * val; else this.g = val;
          } else out[ri] = val;
          off += need; ri++;
        }
        this.off = off; this.ri = 0;
        if (b === M_V) {
          // all of Q, K, V are done (they share the input), nothing else to do here
        }
        this.pc += 3;
        continue;
      }
      // T_HEAD: tied embedding rows, region E; switch stream on first entry
      if (!this.inHead) {
        this.inHead = true; this.blk = 0; this.off = 0; this.ri = 0; this.cur = null;
        this.nk = 0;
        return;
      }
      rows = a; need = 2 + c.d;
      var xh = this.xq, sh = this.xs, rep = this.rep, pen = c.rep, tkI = this.tkI, tkV = this.tkV, K = this.K;
      var d = c.d, sp0 = c.bos, sp1 = c.sep, sp2 = c.pad, nk = this.nk;
      while (ri < rows) {
        if (off + need > B) { this.blk++; this.off = 0; this.ri = ri; this.nk = nk; this.cur = null; return; }
        if (ri !== sp0 && ri !== sp1 && ri !== sp2) {
          acc = 0; o = off + 2;
          for (k = 0; k < d; k++) acc += w[o + k] * xh[k];
          var lg = acc * f16(u[off], u[off + 1]) * sh;
          if (rep[ri]) lg = lg > 0 ? lg / pen : lg * pen;
          if (nk < K || lg > tkV[nk - 1]) {
            var j = nk < K ? nk++ : K - 1;
            while (j > 0 && tkV[j - 1] < lg) { tkV[j] = tkV[j - 1]; tkI[j] = tkI[j - 1]; j--; }
            tkV[j] = lg; tkI[j] = ri;
          }
        }
        off += need; ri++;
      }
      this.nk = nk; this.off = off; this.ri = 0;
      this.pc += 3;
    }
  };

  P.applyNorm = function (b) {
    var c = this.c, nw = this.nw, eps = c.eps, i, h;
    if (b === N_PRE_ATT || b === N_PRE_FFN || b === N_FINAL) {
      rms(this.x, 0, c.d, nw, this.h, eps);
      this.xs = quant(this.h, c.d, this.xq);
    } else if (b === N_Q) {
      for (h = 0; h < c.Hq; h++) rms(this.q, h * c.hd, c.hd, nw, this.q, eps);
    } else if (b === N_K) {
      rms(this.k, 0, c.hd, nw, this.k, eps);
    } else { // post-attn / post-ffn: o = norm(o) * w ; x += o
      rms(this.o, 0, c.d, nw, this.o, eps);
      var x = this.x, o = this.o;
      for (i = 0; i < c.d; i++) x[i] += o[i];
      if (b === N_POST_FFN) this.lay++;
    }
  };

  function rope(v, o, half, cs, sn) {
    for (var i = 0; i < half; i++) {
      var a = v[o + i], b = v[o + i + half];
      v[o + i] = a * cs[i] - b * sn[i];
      v[o + i + half] = b * cs[i] + a * sn[i];
    }
  }

  P.attention = function () {
    var c = this.c, hd = c.hd, half = hd >> 1, pos = this.pos, l = this.lay;
    var cs = this.cs, sn = this.sn, q = this.q, i, t, h, acc;
    for (h = 0; h < c.Hq; h++) rope(q, h * hd, half, cs, sn);
    rope(this.k, 0, half, cs, sn);
    var base = l * c.ctx, kc = this.kc, vc = this.vc;
    var tmp = this.qq;
    this.ks[base + pos] = quant(this.k, hd, tmp);
    for (i = 0; i < hd; i++) kc[(base + pos) * hd + i] = tmp[i];
    this.vs[base + pos] = quant(this.v, hd, tmp);
    for (i = 0; i < hd; i++) vc[(base + pos) * hd + i] = tmp[i];
    var sc = this.sc, wi = this.wi, ks = this.ks, vs = this.vs, att = this.att;
    var scale = 1 / Math.sqrt(hd), qq = this.qq;
    for (h = 0; h < c.Hq; h++) {
      var qs = 0, mx = -1e30;
      // quantize this head's query
      var hm = 0;
      for (i = 0; i < hd; i++) { var av = q[h * hd + i]; if (av < 0) av = -av; if (av > hm) hm = av; }
      if (hm > 0) { qs = hm / 127; for (i = 0; i < hd; i++) qq[i] = Math.round(q[h * hd + i] / qs); }
      else for (i = 0; i < hd; i++) qq[i] = 0;
      for (t = 0; t <= pos; t++) {
        acc = 0;
        var ko = (base + t) * hd;
        for (i = 0; i < hd; i++) acc += qq[i] * kc[ko + i];
        var s = acc * qs * ks[base + t] * scale;
        sc[t] = s;
        if (s > mx) mx = s;
      }
      var sum = 0, wmax = 0;
      for (t = 0; t <= pos; t++) { var p = Math.exp(sc[t] - mx); sum += p; sc[t] = p; }
      for (t = 0; t <= pos; t++) { var wv = sc[t] / sum * vs[base + t]; sc[t] = wv; if (wv > wmax) wmax = wv; }
      if (wmax > 0) for (t = 0; t <= pos; t++) wi[t] = Math.round(sc[t] / wmax * 511);
      else for (t = 0; t <= pos; t++) wi[t] = 0;
      var f = wmax / 511;
      for (i = 0; i < hd; i++) {
        acc = 0;
        var vo = base * hd + i;
        for (t = 0; t <= pos; t++) acc += wi[t] * vc[vo + t * hd];
        att[h * hd + i] = acc * f;
      }
    }
    this.xs = quant(att, c.Hq * hd, this.xq);
  };

  P.finish = function () {
    this.done = true; this.inHead = false;
    this.cur = null; this.cur8 = null; // let the GC reclaim the last block now
    if (this.end !== this.ops.length) return;
    var c = this.c, nk = this.nk, tkI = this.tkI, tkV = this.tkV;
    if (nk === 0) { this.out = c.eos; return; }
    var temp = c.temp;
    if (!(temp > 0)) { this.out = tkI[0]; return; }
    var sum = 0, i, ps = [];
    for (i = 0; i < nk; i++) { var p = Math.exp((tkV[i] - tkV[0]) / temp); ps.push(p); sum += p; }
    var r = Math.random() * sum;
    for (i = 0; i < nk; i++) { r -= ps[i]; if (r <= 0) break; }
    this.out = tkI[i < nk ? i : nk - 1];
  };

  /** Mark a token as "recently used" for the repetition penalty. */
  P.seen = function (tok) { this.rep[tok] = 1; };
  P.clearSeen = function () { var r = this.rep; for (var i = 0; i < r.length; i++) r[i] = 0; };

  /* ---------------- tokenizer (vocab file) ----------------
   * vocab file: u16 V | u16 off[V+1] (id order, into pool) | u16 sorted[V] | pool bytes
   * "sorted" lists ids ordered by token bytes (specials excluded -> 0xFFFF padding at the end).
   */
  function Tok(u8) {
    this.u = u8;
    this.V = u8[0] | (u8[1] << 8);
    this.oo = 2;                       // offsets start
    this.so = 2 + 2 * (this.V + 1);    // sorted start
    this.po = this.so + 2 * this.V;    // pool start
    var n = 0, ml = 1;
    for (var i = 0; i < this.V; i++) {
      var id = this.rd(this.so + 2 * i);
      if (id === 0xFFFF) break;
      n++;
      var L = this.len(id);
      if (L > ml) ml = L;
    }
    this.ns = n; this.ml = ml;
  }
  Tok.prototype.rd = function (o) { return this.u[o] | (this.u[o + 1] << 8); };
  Tok.prototype.start = function (id) { return this.rd(this.oo + 2 * id); };
  Tok.prototype.len = function (id) { return this.rd(this.oo + 2 * id + 2) - this.rd(this.oo + 2 * id); };
  // compare token `id` bytes with s[p .. p+n)
  Tok.prototype.cmp = function (id, s, p, n) {
    var u = this.u, o = this.po + this.start(id), L = this.len(id), m = L < n ? L : n;
    for (var i = 0; i < m; i++) {
      var a = u[o + i], b = s.charCodeAt(p + i);
      if (a !== b) return a < b ? -1 : 1;
    }
    return L === n ? 0 : (L < n ? -1 : 1);
  };
  Tok.prototype.find = function (s, p, n) {
    var lo = 0, hi = this.ns - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1, id = this.rd(this.so + 2 * mid), r = this.cmp(id, s, p, n);
      if (r === 0) return id;
      if (r < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  };
  /** Greedy longest-match encoding (same algorithm as tools/gemma_nano/vocab.py). */
  Tok.prototype.encode = function (s) {
    var out = [], p = 0, N = s.length;
    while (p < N) {
      var n = Math.min(this.ml, N - p), id = -1;
      while (n > 0) { id = this.find(s, p, n); if (id >= 0) break; n--; }
      if (id < 0) { p++; continue; }
      out.push(id); p += n;
    }
    return out;
  };
  Tok.prototype.decode = function (id) {
    var u = this.u, o = this.po + this.start(id), L = this.len(id), s = '';
    for (var i = 0; i < L; i++) s += String.fromCharCode(u[o + i]);
    return s;
  };

  /** Normalise user text exactly like vocab.normalize_prompt() in Python. */
  function normPrompt(s) {
    var out = '', sp = true;
    for (var i = 0; i < s.length; i++) {
      var ch = s.charCodeAt(i);
      if (ch >= 65 && ch <= 90) ch += 32;
      var ok = (ch >= 97 && ch <= 122) || (ch >= 48 && ch <= 57) || ch === 63 || ch === 33 ||
        ch === 46 || ch === 44 || ch === 39;
      if (ok) { out += String.fromCharCode(ch); sp = false; }
      else if (!sp) { out += ' '; sp = true; }
    }
    if (out.length && out.charCodeAt(out.length - 1) === 32) out = out.substring(0, out.length - 1);
    return out;
  }

  return { Engine: Engine, Tok: Tok, f16: f16, normPrompt: normPrompt };
})();

export default GN;

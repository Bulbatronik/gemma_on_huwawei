/*
 * Generation page: tokenises the prompt, runs the model block-by-block from
 * internal://app/ files and streams the answer onto the screen.
 * All engine state lives in module variables (not in `data`) to keep the
 * reactive view-model small.
 */
import file from '@system.file';
import router from '@system.router';
import brightness from '@system.brightness';
import GN from '../../common/gn_engine.js';
import META from '../../common/gn_meta.js';

var S = null;

function pad3(n) {
  return n < 10 ? '00' + n : (n < 100 ? '0' + n : '' + n);
}

function keepOn(on) {
  try {
    brightness.setKeepScreenOn({ keepScreenOn: on });
  } catch (e) {
    console.info('keepScreenOn unsupported');
  }
}

function bytesToStr(u) {
  var s = '';
  for (var i = 0; i < u.length; i++) s += String.fromCharCode(u[i]);
  return s;
}

export default {
  data: {
    prompt: '',
    answer: '',
    status: 'Loading...',
    btn: 'Stop'
  },
  onInit() {
    S = { run: true, vm: this };
    keepOn(true);
    var vm = this;
    file.readText({
      uri: 'internal://app/prompt',
      success: function (d) {
        vm.prompt = d.text;
        vm.loadVocab(GN.normPrompt(d.text));
      },
      fail: function (d, code) { vm.fail('no prompt (' + code + ')'); }
    });
  },
  onDestroy() {
    if (S) S.run = false;
    S = null;
    keepOn(false);
  },
  fail(msg) {
    if (S) S.run = false;
    this.status = msg;
    this.btn = 'New';
    keepOn(false);
  },
  loadVocab(text) {
    var vm = this;
    vm.status = 'Tokenizing...';
    file.readArrayBuffer({
      uri: 'internal://app/voc',
      position: 0,
      length: META.vocabBytes,
      success: function (d) {
        var tok = new GN.Tok(d.buffer);
        var ids = tok.encode(text);
        // keep only the id -> pool offset table for decoding; drop the rest
        var V = tok.V, offs = new Uint16Array(V + 1), u = d.buffer;
        for (var i = 0; i <= V; i++) offs[i] = u[2 + 2 * i] | (u[3 + 2 * i] << 8);
        S.offs = offs;
        S.po = tok.po;
        tok = null;
        u = null;
        var maxP = META.ctx - 12;
        if (ids.length > maxP) ids = ids.slice(ids.length - maxP);
        S.seq = [META.bos].concat(ids).concat([META.sep]);
        vm.startModel();
      },
      fail: function (d, code) { vm.fail('vocab missing (' + code + '). Install model.'); }
    });
  },
  startModel() {
    try {
      S.eng = new GN.Engine(META);
    } catch (e) {
      this.fail('init failed: ' + e.message);
      return;
    }
    S.pos = 0;
    S.i = 0;
    S.nGen = 0;
    S.t0 = Date.now();
    var vm = this;
    S.after = function () { vm.tokenDone(); };
    this.nextToken();
  },
  // decide what to run next: a prompt token (prefill) or a generated token
  nextToken() {
    if (!S || !S.run) return;
    var seq = S.seq;
    if (S.i < seq.length - 1) {
      this.status = 'Reading prompt ' + (S.i + 1) + '/' + (seq.length - 1);
      S.eng.start(seq[S.i], S.pos, false);
    } else {
      if (S.nGen === 0) S.tg = Date.now();
      S.eng.start(S.cur === undefined ? seq[seq.length - 1] : S.cur, S.pos, true);
    }
    S.pos++;
    this.pump();
  },
  pump() {
    var vm = this;
    while (S && S.run) {
      var b = S.eng.next();
      if (b < 0) {
        setTimeout(S.after, 0); // let the UI refresh between tokens
        return;
      }
      S.sync = true;
      S.got = null;
      file.readArrayBuffer({
        uri: 'internal://app/w' + pad3(b),
        position: 0,
        length: META.B,
        success: function (d) {
          if (!S) return;
          if (S.sync) { S.got = d.buffer; return; }
          try { S.eng.feed(d.buffer); } catch (e) { vm.fail('error: ' + e.message); return; }
          vm.pump();
        },
        fail: function (d, code) { vm.fail('read w' + pad3(b) + ' failed (' + code + ')'); }
      });
      if (S.got === null) { S.sync = false; return; } // async: the callback continues
      try { S.eng.feed(S.got); } catch (e) { this.fail('error: ' + e.message); return; }
      S.got = null;
    }
  },
  tokenDone() {
    if (!S || !S.run) return;
    if (S.i < S.seq.length - 1) {
      S.i++;
      if (S.i === S.seq.length - 1) this.status = 'Thinking...';
      this.nextToken();
      return;
    }
    var t = S.eng.out;
    S.nGen++;
    var dt = (Date.now() - S.tg) / S.nGen / 1000;
    this.status = S.nGen + ' tokens, ' + (Math.round(dt * 10) / 10) + ' s/token';
    if (t === META.eos || S.pos >= META.ctx) {
      this.done();
      return;
    }
    S.eng.seen(t);
    S.cur = t;
    this.appendToken(t);
  },
  appendToken(t) {
    var vm = this, a = S.offs[t], n = S.offs[t + 1] - a;
    if (n <= 0) { vm.nextToken(); return; }
    file.readArrayBuffer({
      uri: 'internal://app/voc',
      position: S.po + a,
      length: n,
      success: function (d) {
        var s = bytesToStr(d.buffer);
        vm.answer = vm.answer.length === 0 && s.charAt(0) === ' ' ? s.substring(1) : vm.answer + s;
        vm.nextToken();
      },
      fail: function (d, code) { vm.fail('vocab read failed (' + code + ')'); }
    });
  },
  done() {
    if (S) S.run = false;
    this.btn = 'New';
    this.status = this.status + ' - done';
    keepOn(false);
  },
  main() {
    if (this.btn === 'Stop') {
      if (S) S.run = false;
      this.btn = 'New';
      this.status = 'Stopped. ' + this.status;
      keepOn(false);
      return;
    }
    router.replace({ uri: 'pages/kb/index' });
  },
  menu() {
    if (S) S.run = false;
    router.replace({ uri: 'pages/index/index' });
  }
}

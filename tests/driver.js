/* Test driver for gn_engine.js.  Runs under the patched desktop JerryScript
 * (globals: readBlock, print) and under Node (see run_node.js, which provides
 * the same globals).  Expects the global ARGS = {dir, prompts:[...], maxNew}.
 * Prints one line per prompt:  OUT <id> <id> ...  and a timing line. */
function bytesToStr(u) {
  var s = '';
  for (var i = 0; i < u.length; i++) s += String.fromCharCode(u[i]);
  return s;
}

function runAll() {
  var dir = ARGS.dir;
  var meta = JSON.parse(bytesToStr(readBlock(dir + '/meta.json')));
  meta.temp = 0; // greedy for reproducible tests
  var tok = new GN.Tok(readBlock(dir + '/vocab.bin'));
  var ids = [];
  for (var p = 0; p < ARGS.prompts.length; p++) {
    var ptxt = GN.normPrompt(ARGS.prompts[p]);
    ids.push([meta.bos].concat(tok.encode(ptxt)).concat([meta.sep]));
  }
  tok = null; // release the vocab like the watch does
  var eng = new GN.Engine(meta);
  var t0 = Date.now(), nTok = 0, nBlk = 0;
  for (p = 0; p < ids.length; p++) {
    var seq = ids[p], out = [], pos = 0, i, b;
    eng.clearSeen();
    for (i = 0; i < seq.length - 1; i++) {
      eng.start(seq[i], pos++, false);
      while ((b = eng.next()) >= 0) { eng.feed(readBlock(dir + '/model.bin', b * meta.B, meta.B)); nBlk++; }
      nTok++;
    }
    var cur = seq[seq.length - 1];
    while (pos < meta.ctx && out.length < ARGS.maxNew) {
      eng.start(cur, pos++, true);
      while ((b = eng.next()) >= 0) { eng.feed(readBlock(dir + '/model.bin', b * meta.B, meta.B)); nBlk++; }
      nTok++;
      cur = eng.out;
      out.push(cur);
      eng.seen(cur);
      if (cur === meta.eos) break;
    }
    print('PROMPT ' + seq.join(' '));
    print('OUT ' + out.join(' '));
  }
  var dt = Date.now() - t0;
  print('TIME ' + dt + ' ms ' + nTok + ' tokens ' + nBlk + ' blocks');
}

runAll();

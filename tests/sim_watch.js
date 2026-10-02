/*
 * Runs the real watch pages (index -> installer pages -> kb -> gen) in Node
 * with mocked @system.file / @system.router / @system.brightness.
 *
 *   node tests/sim_watch.js "<prompt>" [--sync] [--temp0]
 *
 * --sync   : file callbacks fire synchronously (tests the gen page trampoline)
 * --temp0  : greedy decoding, so the answer can be compared with the NumPy reference
 * Prints "ANSWER: <text>" on success.
 */
var fs = require('fs');
var path = require('path');
var ROOT = path.join(__dirname, '..', 'watch/entry/src/main/js/MainAbility');
var args = process.argv.slice(2);
var PROMPT = args[0] || 'hello';
var SYNC = args.indexOf('--sync') >= 0;
var TEMP0 = args.indexOf('--temp0') >= 0;

var FS = {};            // internal://app/<name> -> Buffer
var stats = { reads: 0, writes: 0, pages: 0 };
function later(fn) { if (SYNC) fn(); else setImmediate(fn); }
function key(uri) {
  if (uri.indexOf('internal://app/') !== 0) throw new Error('bad uri ' + uri);
  return uri.substring(15);
}

var DIRS = {};
function noDir(k) {
  var i = k.lastIndexOf('/');
  return i >= 0 ? !DIRS[k.substring(0, i)] : !DIRS[''];
}
var file = {
  mkdir: function (o) {
    var k = key(o.uri + '/').replace(/\/$/, '');
    if (!o.recursive && k.indexOf('/') >= 0 && !DIRS[k.substring(0, k.lastIndexOf('/'))]) {
      return later(function () { o.fail && o.fail('no parent', 301); });
    }
    DIRS[''] = true;
    DIRS[k] = true;
    later(function () { o.success && o.success(); });
  },
  writeArrayBuffer: function (o) {
    if (noDir(key(o.uri))) return later(function () { o.fail && o.fail('no dir', 301); });
    if (!(o.buffer instanceof Uint8Array)) throw new Error('writeArrayBuffer needs Uint8Array');
    stats.writes++;
    var k = key(o.uri), old = FS[k] || Buffer.alloc(0), pos = o.position || 0;
    var nb = Buffer.alloc(Math.max(old.length, pos + o.buffer.length));
    old.copy(nb);
    Buffer.from(o.buffer).copy(nb, pos);
    FS[k] = nb;
    later(function () { o.success && o.success(); });
  },
  readArrayBuffer: function (o) {
    stats.reads++;
    var b = FS[key(o.uri)];
    if (!b) return later(function () { o.fail && o.fail('no file', 301); });
    var pos = o.position || 0, len = o.length || (b.length - pos);
    var s = b.subarray(pos, pos + len);
    var u = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) u[i] = s[i];
    later(function () { o.success({ buffer: u }); });
  },
  writeText: function (o) {
    var k = key(o.uri);
    if (noDir(k)) return later(function () { o.fail && o.fail('no dir', 301); });
    FS[k] = o.append && FS[k] ? Buffer.concat([FS[k], Buffer.from(o.text)]) : Buffer.from(o.text);
    later(function () { o.success && o.success(); });
  },
  readText: function (o) {
    var b = FS[key(o.uri)];
    later(function () { if (b) o.success({ text: b.toString() }); else o.fail && o.fail('no file', 301); });
  }
};

var current = null, onPage = null;
var router = { replace: function (o) { setImmediate(function () { load(o.uri); }); } };
var brightness = { setKeepScreenOn: function () {} };
var MODS = {};

function compile(rel) {
  var abs = path.join(ROOT, rel);
  if (MODS[abs]) return MODS[abs];
  var src = fs.readFileSync(abs, 'utf8');
  var imports = {};
  var out = src.split('\n').map(function (line) {
    var m = line.match(/^import (\w+) from '([^']+)';/);
    if (!m) return line;
    imports[m[1]] = m[2];
    return '';
  }).join('\n');
  var ix = out.lastIndexOf('export default');
  out = out.substring(0, ix) + 'return ' + out.substring(ix + 14);
  var names = Object.keys(imports);
  var vals = names.map(function (n) {
    var spec = imports[n];
    if (spec === '@system.file') return file;
    if (spec === '@system.router') return router;
    if (spec === '@system.brightness') return brightness;
    return compile(path.relative(ROOT, path.join(path.dirname(abs), spec)));
  });
  var mod = Function.apply(null, names.concat([out])).apply(null, vals);
  if (rel === 'common/gn_meta.js' && TEMP0) mod.temp = 0;
  MODS[abs] = mod;
  return mod;
}

function load(uri) {
  stats.pages++;
  var def = compile(uri + '.js');
  var vm = {};
  var data = def.data || {};
  for (var k in data) vm[k] = data[k];
  for (k in def) if (typeof def[k] === 'function') vm[k] = def[k];
  current = { uri: uri, vm: vm };
  if (vm.onInit) vm.onInit();
  if (vm.onReady) vm.onReady();
  if (onPage) onPage(uri, vm);
}

// ---- script the user's taps ----
var phase = 'start', t0 = Date.now();
onPage = function (uri, vm) {
  if (uri === 'pages/index/index' && phase === 'start') {
    phase = 'installing';
    setImmediate(function () { vm.goInstall(); });
  } else if (uri === 'pages/index/index' && phase === 'installing') {
    phase = 'chat';
    setTimeout(function () {
      if (!vm.ready) { console.log('FAIL: model not ready after install: ' + vm.status); process.exit(1); }
      vm.goChat();
    }, 5);
  } else if (uri === 'pages/kb/index') {
    for (var i = 0; i < PROMPT.length; i++) {
      var ch = PROMPT.charAt(i);
      if (ch === ' ') vm.sp(); else vm.add(ch);
    }
    vm.go();
  } else if (uri === 'pages/gen/index') {
    var iv = setInterval(function () {
      if (vm.btn === 'New') {
        clearInterval(iv);
        console.log('STATUS: ' + vm.status);
        console.log('ANSWER: ' + vm.answer);
        console.log('SCREEN: ' + JSON.stringify([vm.p0, vm.p1, vm.a0, vm.a1, vm.a2, vm.a3, vm.a4, vm.a5, vm.a6]));
        console.log('stats: ' + JSON.stringify(stats) + ' in ' + (Date.now() - t0) + ' ms');
        process.exit(vm.status.indexOf('done') >= 0 ? 0 : 1);
      }
    }, 2);
  }
};
load('pages/index/index');

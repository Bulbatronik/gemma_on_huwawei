/*
 * Model installer helper used by the generated pages/iNNN pages.
 *
 * Why pages: Lite Wearable JS cannot read files shipped in the package
 * (rawfile), only files it wrote itself under internal://app/.  Each page
 * carries a few weight blocks as base64 string literals (a page's compiled JS
 * must stay well under the ~48 KB per-page limit and the 48 KB heap), decodes
 * them and writes them with writeArrayBuffer, then routes to the next page.
 */
import file from '@system.file';
import router from '@system.router';

var TAB = null;

function table() {
  if (TAB) return TAB;
  var A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  TAB = new Uint8Array(128);
  for (var i = 0; i < 64; i++) TAB[A.charCodeAt(i)] = i;
  return TAB;
}

function b64(s) {
  var T = table(), n = s.length, pad = 0;
  if (n > 0 && s.charCodeAt(n - 1) === 61) pad++;
  if (n > 1 && s.charCodeAt(n - 2) === 61) pad++;
  var out = new Uint8Array((n / 4) * 3 - pad), o = 0;
  for (var i = 0; i < n; i += 4) {
    var v = (T[s.charCodeAt(i)] << 18) | (T[s.charCodeAt(i + 1)] << 12) |
      (T[s.charCodeAt(i + 2)] << 6) | T[s.charCodeAt(i + 3)];
    if (o < out.length) out[o++] = (v >> 16) & 255;
    if (o < out.length) out[o++] = (v >> 8) & 255;
    if (o < out.length) out[o++] = v & 255;
  }
  return out;
}

export default {
  b64: b64,
  // items: [[fileName, base64], ...];  next: page uri or '' for the last page
  run: function (vm, page, pages, items, next, modelId) {
    var i = 0;
    vm.msg = 'Installing ' + page + ' / ' + pages;
    function done() {
      if (next) {
        router.replace({ uri: next });
        return;
      }
      file.writeText({
        uri: 'internal://app/ok',
        text: modelId,
        success: function () { router.replace({ uri: 'pages/index/index' }); },
        fail: function (d, code) { vm.msg = 'marker write failed ' + code; }
      });
    }
    function step() {
      if (i >= items.length) {
        done();
        return;
      }
      var it = items[i], buf = b64(it[1]);
      file.writeArrayBuffer({
        uri: 'internal://app/' + it[0],
        buffer: buf,
        position: 0,
        success: function () {
          buf = null;
          i++;
          setTimeout(step, 0);
        },
        fail: function (d, code) { vm.msg = 'write ' + it[0] + ' failed: ' + code; }
      });
    }
    step();
  }
};

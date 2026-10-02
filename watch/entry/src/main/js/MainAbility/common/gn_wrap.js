/*
 * Manual word wrap: the Lite Wearable compiler only accepts text-overflow
 * clip|ellipsis, so multi-line text is shown as fixed single-line <text> slots.
 */
function wrap(text, width) {
  var lines = [], paras = String(text).split('\n');
  for (var p = 0; p < paras.length; p++) {
    var words = paras[p].split(' '), cur = '';
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      while (w.length > width) {            // hard-split very long words
        if (cur.length) { lines.push(cur); cur = ''; }
        lines.push(w.substring(0, width));
        w = w.substring(width);
      }
      if (!cur.length) cur = w;
      else if (cur.length + 1 + w.length <= width) cur = cur + ' ' + w;
      else { lines.push(cur); cur = w; }
    }
    lines.push(cur);
  }
  return lines;
}

/* Put `text` into vm[prefix + 0 .. n-1]; tail=true shows the last n lines. */
function fill(vm, prefix, n, text, width, tail) {
  var lines = wrap(text, width), start = 0;
  if (tail && lines.length > n) start = lines.length - n;
  for (var i = 0; i < n; i++) {
    var s = lines[start + i];
    vm[prefix + i] = s === undefined ? '' : s;
  }
}

export default { wrap: wrap, fill: fill };

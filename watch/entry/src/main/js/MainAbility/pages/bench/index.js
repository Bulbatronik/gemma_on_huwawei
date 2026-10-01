/*
 * Measures what the inference runtime depends on:
 *   Speed: int8 multiply-accumulate rate of the JS engine and the latency of
 *          @system.file.readArrayBuffer for 1/2/4/8 KB blocks.
 *   Heap:  free JS heap.  Running out of heap aborts the app (it cannot be
 *          caught), so the probe writes its progress to a file before every
 *          2 KB step; after a crash, reopening this page shows the result.
 * Results are also saved to internal://app/bench.txt.
 */
import file from '@system.file';
import router from '@system.router';

var hold = null;

function macTest() {
  var R = 64, C = 128, w = new Int8Array(R * C), x = new Int8Array(C), y = new Float32Array(R);
  var i, r, k, o, a;
  for (i = 0; i < R * C; i++) w[i] = (i * 37) % 255 - 127;
  for (i = 0; i < C; i++) x[i] = (i * 11) % 255 - 127;
  var t0 = Date.now(), n = 0;
  while (Date.now() - t0 < 1500) {
    o = 0;
    for (r = 0; r < R; r++) {
      a = 0;
      for (k = 0; k < C; k++) a += w[o + k] * x[k];
      o += C;
      y[r] = a * 0.01;
    }
    n++;
  }
  return n * R * C / ((Date.now() - t0) / 1000);
}

export default {
  data: {
    out: 'Speed: compute + flash read test (~15 s).\nHeap: free memory test (may close the app; reopen this page to see the result).'
  },
  onInit() {
    var vm = this;
    file.readText({
      uri: 'internal://app/heap',
      success: function (d) {
        var t = d.text;
        if (t.indexOf('probing') === 0) {
          vm.out = 'Last heap test stopped the app at ~' + t.substring(8) + ' KB, so about ' +
            t.substring(8) + ' KB of JS heap is free on this page.';
        } else if (t.length > 0) {
          vm.out = 'Last heap test: ' + t;
        }
      }
    });
  },
  save(text) {
    file.writeText({ uri: 'internal://app/bench.txt', text: text, append: true });
  },
  speed() {
    var vm = this;
    vm.out = 'Running compute test...';
    setTimeout(function () {
      var macs = macTest();
      var res = 'int8 MAC: ' + (Math.round(macs / 10000) / 100) + ' M/s';
      vm.out = res + '\nflash test...';
      var sizes = [1024, 2048, 4096, 8192], si = 0;
      function nextSize() {
        if (si >= sizes.length) {
          vm.out = res;
          vm.save(res + '\n');
          return;
        }
        var n = sizes[si], buf = new Uint8Array(n);
        for (var i = 0; i < n; i++) buf[i] = i & 255;
        file.writeArrayBuffer({
          uri: 'internal://app/bt' + n,
          buffer: buf,
          success: function () {
            buf = null;
            var k = 0, reps = 20, t0 = Date.now();
            function rd() {
              if (k >= reps) {
                var ms = (Date.now() - t0) / reps;
                res = res + '\nread ' + (n / 1024) + 'KB: ' + (Math.round(ms * 10) / 10) + ' ms';
                vm.out = res;
                si++;
                setTimeout(nextSize, 0);
                return;
              }
              k++;
              file.readArrayBuffer({
                uri: 'internal://app/bt' + n,
                position: 0,
                length: n,
                success: function () { rd(); },
                fail: function (d, code) { vm.out = res + '\nread failed ' + code; }
              });
            }
            rd();
          },
          fail: function (d, code) { vm.out = res + '\nwrite failed ' + code; }
        });
      }
      nextSize();
    }, 50);
  },
  heap() {
    var vm = this, kb = 0, LIMIT = 200;
    hold = [];
    function step() {
      if (kb >= LIMIT) {
        hold = null;
        var msg = 'more than ' + LIMIT + ' KB free';
        file.writeText({ uri: 'internal://app/heap', text: msg });
        vm.out = 'Heap: ' + msg;
        return;
      }
      file.writeText({
        uri: 'internal://app/heap',
        text: 'probing ' + kb,
        success: function () {
          hold.push(new Uint8Array(2048));
          kb += 2;
          vm.out = 'Heap test: ' + kb + ' KB allocated...';
          setTimeout(step, 0);
        },
        fail: function (d, code) { vm.out = 'cannot write probe file ' + code; }
      });
    }
    step();
  },
  menu() {
    hold = null;
    router.replace({ uri: 'pages/index/index' });
  }
}

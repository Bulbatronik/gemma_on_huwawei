import file from '@system.file';
import F from '../../common/gn_fs.js';
import router from '@system.router';
import META from '../../common/gn_meta.js';
import W from '../../common/gn_wrap.js';

export default {
  data: {
    status: '',
    s0: 'Checking model...',
    s1: '',
    installLabel: 'Install model',
    foot: '',
    ready: false
  },
  onInit() {
    this.foot = META.name + ' ' + Math.round(META.params / 1000) + 'K params';
    var vm = this;
    file.readText({
      uri: F.P + 'ok',
      success: function (d) {
        if (d.text === META.modelId) {
          vm.ready = true;
          vm.say('Model installed. Runs 100% on this watch.');
          vm.installLabel = 'Reinstall model';
        } else {
          vm.say('A different model is installed. Install this one first.');
        }
      },
      fail: function () {
        vm.say('Model not installed yet (one-time, ~' + Math.round(META.nBlocks * META.B / 1024) + ' KB).');
      }
    });
  },
  say(t) {
    this.status = t;
    W.fill(this, 's', 2, t, 32, false);
  },
  goChat() {
    if (!this.ready) {
      this.say('Install the model first.');
      return;
    }
    router.replace({ uri: 'pages/kb/index' });
  },
  goInstall() {
    router.replace({ uri: 'pages/i000/index' });
  },
  goBench() {
    router.replace({ uri: 'pages/bench/index' });
  }
}

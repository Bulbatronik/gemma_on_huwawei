import file from '@system.file';
import router from '@system.router';
import META from '../../common/gn_meta.js';

export default {
  data: {
    status: 'Checking model...',
    installLabel: 'Install model',
    foot: '',
    ready: false
  },
  onInit() {
    this.foot = META.name + ' ' + Math.round(META.params / 1000) + 'K params';
    var vm = this;
    file.readText({
      uri: 'internal://app/ok',
      success: function (d) {
        if (d.text === META.modelId) {
          vm.ready = true;
          vm.status = 'Model installed. Runs 100% on this watch.';
          vm.installLabel = 'Reinstall model';
        } else {
          vm.status = 'A different model is installed. Install this one first.';
        }
      },
      fail: function () {
        vm.status = 'Model not installed yet (one-time, ~' + Math.round(META.nBlocks * META.B / 1024) + ' KB).';
      }
    });
  },
  goChat() {
    if (!this.ready) {
      this.status = 'Install the model first.';
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

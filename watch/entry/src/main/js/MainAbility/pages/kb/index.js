import file from '@system.file';
import router from '@system.router';

var KEYS = 'abcdefghijklmnopqrstuvwxyz,?';
var IDEAS = ['hello', 'how are you', 'tell me a joke about cats', 'what is the sun',
  'why is the sky blue', 'how do i sleep better', 'give me a fun fact about space', 'what is a dog',
  'tell me something nice', 'what is coffee', 'how do i make tea', 'what is a robot'];
var MAXLEN = 60;

export default {
  data: {
    text: '',
    view: 'type a question',
    ideaIdx: 0
  },
  render() {
    var t = this.text;
    if (t.length === 0) { this.view = 'type a question'; return; }
    if (t.length > 22) t = '...' + t.substring(t.length - 21);
    this.view = t + '_';
  },
  add(ch) {
    if (this.text.length >= MAXLEN) return;
    this.text = this.text + ch;
    this.render();
  },
  k0() { this.add(KEYS.charAt(0)); },
  k1() { this.add(KEYS.charAt(1)); },
  k2() { this.add(KEYS.charAt(2)); },
  k3() { this.add(KEYS.charAt(3)); },
  k4() { this.add(KEYS.charAt(4)); },
  k5() { this.add(KEYS.charAt(5)); },
  k6() { this.add(KEYS.charAt(6)); },
  k7() { this.add(KEYS.charAt(7)); },
  k8() { this.add(KEYS.charAt(8)); },
  k9() { this.add(KEYS.charAt(9)); },
  k10() { this.add(KEYS.charAt(10)); },
  k11() { this.add(KEYS.charAt(11)); },
  k12() { this.add(KEYS.charAt(12)); },
  k13() { this.add(KEYS.charAt(13)); },
  k14() { this.add(KEYS.charAt(14)); },
  k15() { this.add(KEYS.charAt(15)); },
  k16() { this.add(KEYS.charAt(16)); },
  k17() { this.add(KEYS.charAt(17)); },
  k18() { this.add(KEYS.charAt(18)); },
  k19() { this.add(KEYS.charAt(19)); },
  k20() { this.add(KEYS.charAt(20)); },
  k21() { this.add(KEYS.charAt(21)); },
  k22() { this.add(KEYS.charAt(22)); },
  k23() { this.add(KEYS.charAt(23)); },
  k24() { this.add(KEYS.charAt(24)); },
  k25() { this.add(KEYS.charAt(25)); },
  k26() { this.add(KEYS.charAt(26)); },
  k27() { this.add(KEYS.charAt(27)); },
  sp() {
    var t = this.text;
    if (t.length > 0 && t.charAt(t.length - 1) !== ' ') this.add(' ');
  },
  del() {
    if (this.text.length > 0) this.text = this.text.substring(0, this.text.length - 1);
    this.render();
  },
  idea() {
    this.text = IDEAS[this.ideaIdx % IDEAS.length];
    this.ideaIdx = this.ideaIdx + 1;
    this.render();
  },
  back() {
    router.replace({ uri: 'pages/index/index' });
  },
  go() {
    if (this.text.length === 0) return;
    var vm = this;
    vm.view = 'thinking...';
    file.writeText({
      uri: 'internal://app/prompt',
      text: vm.text,
      success: function () { router.replace({ uri: 'pages/gen/index' }); },
      fail: function (d, code) { vm.view = 'save failed ' + code; }
    });
  }
}

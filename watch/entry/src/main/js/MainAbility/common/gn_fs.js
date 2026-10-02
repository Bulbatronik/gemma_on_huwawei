/*
 * All app files live in internal://app/gn/.  On some devices (and the DevEco
 * simulator) the app's data directory does not exist until something creates
 * it, and writing straight into internal://app/ fails with code 301
 * ("file or directory does not exist").  ensure() creates it first.
 */
import file from '@system.file';

var DIR = 'internal://app/gn';

export default {
  P: DIR + '/',
  ensure: function (cb) {
    file.mkdir({
      uri: DIR,
      recursive: true,
      success: function () { cb(0); },
      fail: function (d, code) { cb(code); } // usually "already exists"; later writes report real errors
    });
  }
};

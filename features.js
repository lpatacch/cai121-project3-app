// =====================================================================
//  features.js  —  YOUR APP'S SETTINGS
//  This is the only file you need to change (plus your model folder).
//  Change the words after the colons. Keep the quotes, commas and { }.
//  After you save and publish, reload the page on your phone.
// =====================================================================

const APP = {

  // ---------- About your app ----------
  title: "CHANGE_ME: your app's name",
  purpose: "CHANGE_ME: one sentence. Who is it for, and what does it help them decide?",

  // ---------- Test mode ----------
  // true  = while you test (Project 3 Part 3). Your features are hidden, and
  //         every answer shows its confidence and the time it took.
  // false = the finished app that other people use.
  testMode: true,

  // ---------- "Not sure" ----------
  // From 0 to 1. If the model's confidence is lower than this, the app says
  // "Not sure" (classification) or hides the box (detection).
  // Start at 0 so you see every answer. Raise it later (for example 0.7).
  // (Detection apps never show boxes below 0.25, so for detection 0 works like 0.25.)
  notSure: 0,
  notSureText: "Not sure. Try a closer or clearer view.",

  // =================================================================
  //  FEATURES: turn on 1 or 2 from the menu for YOUR type of app.
  //  To turn one on, change  on: false  to  on: true
  // =================================================================

  // ---------- Classification and detection ----------

  // Alert: show a big message (and vibrate, if the phone can) when one class appears.
  alert: { on: false, className: "CHANGE_ME", text: "CHANGE_ME: what the alert says" },

  // Guide the user: tell them when the picture is too dark or too blurry.
  // Detection apps also say "Move closer" when the objects look very small.
  guide: { on: false, tooDarkBelow: 60, tooBlurryBelow: 40, moveCloserBelow: 0.02 },

  // Best photo: keep the best picture so far (highest confidence), with a Save button.
  bestPhoto: { on: false },

  // ---------- Classification only ----------

  // A message for each class. Use your own class names, exactly as in classes.json.
  messages: {
    on: false,
    text: {
      "CHANGE_ME class name": "CHANGE_ME: what the user should do",
      "CHANGE_ME other class": "CHANGE_ME: what the user should do",
    },
  },

  // Results history: list the last few answers with their confidence.
  history: { on: false, size: 5 },

  // ---------- Detection only ----------

  // Count: count each class in view, and add the counts to a running tally with a log.
  count: { on: false },

  // Only when confident: show a box only after it stays steady for a few frames in a row.
  onlyWhenConfident: { on: false, frames: 5 },

  // Results summary: what the app has seen since you opened it (most of each class at once).
  summary: { on: false },

  // ---------- Extra credit (optional) ----------
  // true = run your own code in my-feature.js
  myFeature: false,
};

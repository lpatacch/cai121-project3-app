// =====================================================================
//  app.js — the engine of the app. You don't need to change this file.
//  Change features.js instead (and my-feature.js for extra credit).
// =====================================================================
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const show = (el, on) => { el.hidden = !on; };
  const pct = (p) => `${Math.round(p * 100)}%`;

  // ---------- state ----------
  let session = null, inputName = "", SET = null, NAMES = [];
  let video = null, stream = null, busy = false, lastSource = null, lastResult = null;
  const timings = [];
  const view = $("view"), ctx = view.getContext("2d");
  const work = document.createElement("canvas"), wctx = work.getContext("2d", { willReadFrequently: true });
  const A = (typeof APP === "object" && APP) ? APP : {};
  const featuresBroken = !(typeof APP === "object" && APP);
  const on = (k) => !!(A[k] && A[k].on) && !A.testMode;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  function setupError(text) { $("setupError").textContent = text; show($("setupError"), true); }

  // ---------- 1. Load settings + model ----------
  async function loadAll() {
    $("title").textContent = A.title || "Project 3 App";
    document.title = A.title || "Project 3 App";
    $("purpose").textContent = A.purpose || "";
    show($("testBanner"), !!A.testMode);
    if (featuresBroken) setupError("features.js has a typo, so the app can't read your settings. Check your last change: every line needs its quotes, commas and { } (compare with the original file).");
    if (/CHANGE_ME/.test(String(A.title) + String(A.purpose))) setupError("features.js still has CHANGE_ME in the title or purpose. Write your app's name and purpose there.");

    const t0 = performance.now();
    try {
      SET = await (await fetch("model/settings.json", { cache: "no-store" })).json();
    } catch (e) { throw new Error("model/settings.json is missing or broken. Copy it from your CAI121_project2_models folder in Drive."); }
    try {
      NAMES = await (await fetch("model/classes.json", { cache: "no-store" })).json();
    } catch (e) { throw new Error("model/classes.json is missing or broken. Copy it from your CAI121_project2_models folder in Drive."); }
    const r = await fetch("model/model.onnx", { cache: "no-store" });
    if (!r.ok) throw new Error("model/model.onnx is missing. Copy it from your CAI121_project2_models folder in Drive.");
    const bytes = new Uint8Array(await r.arrayBuffer());
    ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    inputName = session.inputNames[0];
    const secs = (performance.now() - t0) / 1000;
    $("sLoad").textContent = secs.toFixed(1);
    $("modelInfo").textContent = `Model: ${SET.task}, ${NAMES.length} classes, ${(bytes.length / 1e6).toFixed(1)} MB, ${SET.image_size} × ${SET.image_size} pixels. Loaded in ${secs.toFixed(1)} s.`;
    checkFeatures();
  }

  // Friendly checks on features.js
  function checkFeatures() {
    const clf = SET.task === "classification", warn = [];
    const clfOnly = ["messages", "history"], detOnly = ["count", "onlyWhenConfident", "summary"];
    for (const k of clf ? detOnly : clfOnly) if (A[k] && A[k].on) warn.push(`"${k}" is only for ${clf ? "detection" : "classification"} apps, so it is turned off.`);
    if (A.alert && A.alert.on && !NAMES.includes(A.alert.className)) warn.push(`alert: "${A.alert.className}" is not one of your classes (${NAMES.join(", ")}). Check the spelling.`);
    if (clf && A.messages && A.messages.on) for (const n of Object.keys(A.messages.text || {})) if (!NAMES.includes(n)) warn.push(`messages: "${n}" is not one of your classes. Check the spelling.`);
    if (warn.length) setupError(warn.join(" "));
    if (clf) { for (const k of detOnly) if (A[k]) A[k].on = false; } else { for (const k of clfOnly) if (A[k]) A[k].on = false; }
  }

  // ---------- 2. Prepare a picture the same way as training ----------
  // Classification: padded to a square with black, RGB pixels 0-255, channels last.
  // Detection (YOLO): padded ("letterbox") with gray 114, RGB 0-1, channels first.
  function prepare(src, w, h) {
    const S = SET.image_size, pc = SET.pad_color || (SET.task === "detection" ? [114, 114, 114] : [0, 0, 0]);
    work.width = work.height = S;
    const r = Math.min(S / w, S / h), nw = Math.round(w * r), nh = Math.round(h * r);
    const dx = Math.floor((S - nw) / 2), dy = Math.floor((S - nh) / 2);
    wctx.fillStyle = `rgb(${pc[0]},${pc[1]},${pc[2]})`; wctx.fillRect(0, 0, S, S);
    wctx.imageSmoothingEnabled = true; wctx.imageSmoothingQuality = "high";
    wctx.drawImage(src, dx, dy, nw, nh);
    const px = wctx.getImageData(0, 0, S, S).data, n = S * S, t = new Float32Array(3 * n);
    if (SET.task === "detection") {
      for (let i = 0; i < n; i++) { t[i] = px[4 * i] / 255; t[n + i] = px[4 * i + 1] / 255; t[2 * n + i] = px[4 * i + 2] / 255; }
      return { tensor: new ort.Tensor("float32", t, [1, 3, S, S]), r, dx, dy, px, S };
    }
    for (let i = 0; i < n; i++) { t[3 * i] = px[4 * i]; t[3 * i + 1] = px[4 * i + 1]; t[3 * i + 2] = px[4 * i + 2]; }
    return { tensor: new ort.Tensor("float32", t, [1, S, S, 3]), r, dx, dy, px, S };
  }

  // Brightness and sharpness (Week 9's sharpness check: variance of the Laplacian), on the prepared picture
  function lookCheck(p) {
    const S = p.S, g = new Float32Array(S * S);
    let sum = 0, cnt = 0;
    const x0 = p.dx, x1 = S - p.dx, y0 = p.dy, y1 = S - p.dy;   // skip the padding
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = y * S + x, v = 0.299 * p.px[4 * i] + 0.587 * p.px[4 * i + 1] + 0.114 * p.px[4 * i + 2];
      g[i] = v; if (x >= x0 && x < x1 && y >= y0 && y < y1) { sum += v; cnt++; }
    }
    let m = 0, m2 = 0, k = 0;
    for (let y = Math.max(1, y0 + 1); y < Math.min(S - 1, y1 - 1); y++) for (let x = Math.max(1, x0 + 1); x < Math.min(S - 1, x1 - 1); x++) {
      const i = y * S + x, lap = g[i - 1] + g[i + 1] + g[i - S] + g[i + S] - 4 * g[i];
      m += lap; m2 += lap * lap; k++;
    }
    const mean = m / Math.max(1, k);
    return { brightness: sum / Math.max(1, cnt), sharpness: m2 / Math.max(1, k) - mean * mean };
  }

  // ---------- 3. Run the model ----------
  async function predict(src, w, h, sourceKind) {
    const p = prepare(src, w, h), t0 = performance.now();
    const out = await session.run({ [inputName]: p.tensor });
    const ms = performance.now() - t0;
    timings.push(ms); if (timings.length > 30) timings.shift();
    const data = out[session.outputNames[0]].data;
    const res = { task: SET.task, ms, source: sourceKind, look: lookCheck(p), w, h };
    if (SET.task === "classification") {
      res.scores = Array.from(data, (c, i) => ({ name: NAMES[i] ?? `class ${i}`, confidence: c })).sort((a, b) => b.confidence - a.confidence);
      const top = res.scores[0];
      res.confidence = top.confidence;
      res.answer = top.confidence >= (A.notSure || 0) ? top.name : "Not sure";
    } else {
      // YOLO26 (exported with nms=False) gives its final boxes: x1, y1, x2, y2, score, class
      const all = [];
      for (let i = 0; i + 5 < data.length; i += 6) {
        if (data[i + 4] < 0.05) continue;
        const x1 = (data[i] - p.dx) / p.r, y1 = (data[i + 1] - p.dy) / p.r, x2 = (data[i + 2] - p.dx) / p.r, y2 = (data[i + 3] - p.dy) / p.r;
        all.push({ name: NAMES[data[i + 5] | 0] ?? `class ${data[i + 5] | 0}`, confidence: data[i + 4], x: x1, y: y1, width: x2 - x1, height: y2 - y1 });
      }
      all.sort((a, b) => b.confidence - a.confidence);
      // Test mode shows every box from 25% up. The finished app shows boxes at the "Not sure" level or higher (never below 25%).
      const floor = A.testMode ? 0.25 : Math.max(A.notSure || 0, 0.25);
      res.boxes = all.filter((b) => b.confidence >= floor);
    }
    return res;
  }

  // ---------- 4. Show the answer ----------
  let lastSpoken = "";
  function drawBoxes(boxes, numbered) {
    const w = view.width, lw = Math.max(2, w / 220), fs = Math.max(14, w / 30);
    ctx.lineWidth = lw; ctx.font = `700 ${fs}px system-ui, sans-serif`; ctx.textBaseline = "top";
    boxes.forEach((b, i) => {
      ctx.strokeStyle = css("--box"); ctx.strokeRect(b.x, b.y, b.width, b.height);
      const label = `${numbered ? "#" + (i + 1) + " " : ""}${b.name} ${pct(b.confidence)}`;
      const tw = ctx.measureText(label).width + fs * 0.6, ty = b.y - fs * 1.35 < 0 ? b.y + lw : b.y - fs * 1.35;
      const tx = Math.max(0, Math.min(b.x - lw / 2, w - tw));          // keep the label inside the picture
      ctx.fillStyle = css("--box"); ctx.fillRect(tx, ty, tw, fs * 1.35);
      ctx.fillStyle = "#1E2A36"; ctx.fillText(label, tx + fs * 0.3, ty + fs * 0.18);
    });
  }

  function render(src, res) {
    if (view.width !== res.w || view.height !== res.h) { view.width = res.w; view.height = res.h; }
    ctx.drawImage(src, 0, 0, res.w, res.h);
    let boxes = res.boxes;
    if (res.task === "detection" && on("onlyWhenConfident")) boxes = steadyBoxes(boxes);
    if (res.task === "detection") { drawBoxes(boxes, !!A.testMode); res.shownBoxes = boxes; }

    let text, sub = "";
    if (res.task === "classification") {
      if (res.answer === "Not sure") { text = "Not sure"; sub = A.notSureText || ""; }
      else { text = res.answer; sub = `Confidence ${pct(res.confidence)}`; }
    } else {
      const c = countBy(boxes);
      text = boxes.length ? Object.entries(c).map(([n, k]) => `${k} ${n}`).join(", ") : "Nothing found";
      sub = boxes.length ? `${boxes.length} box${boxes.length > 1 ? "es" : ""}` : (A.notSure ? `No box above ${pct(A.notSure)}` : "");
    }
    $("answerSub").textContent = sub;
    if (text !== lastSpoken) { $("answer").textContent = text; lastSpoken = text; }   // screen readers hear changes only
    if (A.testMode) renderTest(res, boxes); else runFeatures(res, boxes, src);
    if (A.myFeature) runMyFeature(res);
  }

  function countBy(boxes) { const c = {}; boxes.forEach((b) => (c[b.name] = (c[b.name] || 0) + 1)); return c; }

  // Test mode: the numbers you copy into the field-test spreadsheet
  function renderTest(res, boxes) {
    show($("testPanel"), true);
    $("sMs").textContent = Math.round(res.ms);
    $("sAvg").textContent = Math.round(timings.reduce((a, b) => a + b, 0) / timings.length);
    const head = $("testThead"), rows = $("testRows");
    rows.textContent = "";
    if (res.task === "classification") {
      head.innerHTML = '<tr><th scope="col">Rank</th><th scope="col">Class</th><th scope="col">Confidence</th></tr>';
      res.scores.slice(0, 3).forEach((s, i) => addRow(rows, [i + 1, s.name, pct(s.confidence)]));
    } else {
      head.innerHTML = '<tr><th scope="col">Box</th><th scope="col">Class</th><th scope="col">Confidence</th></tr>';
      if (!boxes.length) addRow(rows, ["–", "No boxes at 25% or more", ""]);
      boxes.forEach((b, i) => addRow(rows, ["#" + (i + 1), b.name, pct(b.confidence)]));
    }
  }
  function addRow(tbody, cells) {
    const tr = document.createElement("tr");
    cells.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i === cells.length - 1) td.className = "num"; tr.append(td); });
    tbody.append(tr);
  }

  // ---------- 5. Features ----------
  function runFeatures(res, boxes, src) {
    const clf = res.task === "classification";
    // message for each class
    if (on("messages")) {
      const m = res.answer !== "Not sure" ? (A.messages.text || {})[res.answer] : "";
      $("message").textContent = m || ""; show($("message"), !!m);
    }
    // alert
    if (on("alert")) {
      const hit = clf ? res.answer === A.alert.className : boxes.some((b) => b.name === A.alert.className);
      if (hit && $("alertBox").hidden) { buzz(); }
      $("alertBox").textContent = A.alert.text; show($("alertBox"), hit);
    }
    // guide the user
    if (on("guide")) {
      const g = A.guide, tips = [];
      if (res.look.brightness < (g.tooDarkBelow ?? 60)) tips.push("Too dark: find more light.");
      else if (res.look.sharpness < (g.tooBlurryBelow ?? 40)) tips.push("Too blurry: hold still or move back a little.");
      if (!clf) {
        const big = boxes.reduce((m, b) => Math.max(m, (b.width * b.height) / (res.w * res.h)), 0);
        if (boxes.length && big < (g.moveCloserBelow ?? 0.02)) tips.push("Move closer.");
      }
      if (clf && res.answer === "Not sure" && !tips.length) tips.push("Try again: fill the picture with one object.");
      $("guideBox").textContent = tips.join(" "); show($("guideBox"), tips.length > 0);
    }
    // best photo
    if (on("bestPhoto")) {
      const score = clf ? (res.answer === "Not sure" ? 0 : res.confidence) : boxes.reduce((s, b) => s + b.confidence, 0);
      if (score > best.score) {
        best.score = score;
        const c = document.createElement("canvas"); c.width = view.width; c.height = view.height; c.getContext("2d").drawImage(view, 0, 0);
        best.url = c.toDataURL("image/jpeg", 0.9);
        $("bestImg").src = best.url;
        $("bestText").textContent = clf ? `${res.answer}, confidence ${pct(res.confidence)}` : `${boxes.length} box(es): ${boxes.map((b) => `${b.name} ${pct(b.confidence)}`).join(", ")}`;
        $("bestImg").alt = `Best photo so far: ${$("bestText").textContent}`;
      }
      show($("bestPanel"), best.score > 0);
    }
    // results history (classification)
    if (on("history") && stable(res.answer, res.source)) {
      if (res.answer !== "Not sure") {
        const li = document.createElement("li");
        li.textContent = `${new Date().toLocaleTimeString()}: ${res.answer} (${pct(res.confidence)})`;
        $("historyList").prepend(li);
        while ($("historyList").children.length > (A.history.size || 5)) $("historyList").lastChild.remove();
        show($("historyPanel"), true);
      }
    }
    // count (detection)
    if (on("count")) {
      show($("countPanel"), true);
      currentCounts = countBy(boxes);
      $("countNow").textContent = "In view now: " + (boxes.length ? Object.entries(currentCounts).map(([n, k]) => `${k} ${n}`).join(", ") : "nothing");
    }
    // summary (detection)
    if (on("summary")) {
      show($("summaryPanel"), true);
      summary.answers++;
      const c = countBy(boxes);
      for (const [n, k] of Object.entries(c)) { const s = (summary.cls[n] ||= { most: 0, seen: 0 }); s.most = Math.max(s.most, k); s.seen++; }
      const rows = $("summaryRows"); rows.textContent = "";
      for (const n of NAMES) { const s = summary.cls[n] || { most: 0, seen: 0 }; addRow(rows, [n, s.most, s.seen]); }
      $("summaryNote").textContent = `Answers so far: ${summary.answers}`;
    }
  }

  // "Stable" = the same answer for about 1 second of camera frames (so the history doesn't fill up every frame)
  let stableName = "", stableCount = 0, stableLogged = "";
  function stable(name, source) {
    if (source === "photo") { stableLogged = ""; return true; }
    if (name === stableName) stableCount++; else { stableName = name; stableCount = 1; }
    if (stableCount >= 8 && stableLogged !== name) { stableLogged = name; return true; }
    return false;
  }

  // Only when confident: a class's boxes appear after they stay for N frames in a row
  const streak = {};
  function steadyBoxes(boxes) {
    const need = (A.onlyWhenConfident && A.onlyWhenConfident.frames) || 5, here = new Set(boxes.map((b) => b.name));
    for (const n of NAMES) streak[n] = here.has(n) ? (streak[n] || 0) + 1 : 0;
    if (lastResult && lastResult.source === "photo") return boxes;
    return boxes.filter((b) => streak[b.name] >= need);
  }

  const best = { score: 0, url: "" };
  $("bestSave").addEventListener("click", () => { if (!best.url) return; const a = document.createElement("a"); a.href = best.url; a.download = "best-photo.jpg"; a.click(); });
  $("bestReset").addEventListener("click", () => { best.score = 0; best.url = ""; show($("bestPanel"), false); });

  let currentCounts = {}; const tally = {};
  $("tallyAdd").addEventListener("click", () => {
    const parts = Object.entries(currentCounts);
    for (const [n, k] of parts) tally[n] = (tally[n] || 0) + k;
    const li = document.createElement("li");
    li.textContent = `${new Date().toLocaleTimeString()}: added ${parts.length ? parts.map(([n, k]) => `${k} ${n}`).join(", ") : "nothing"}`;
    $("tallyLog").prepend(li); drawTally();
  });
  $("tallyReset").addEventListener("click", () => { for (const k in tally) delete tally[k]; $("tallyLog").textContent = ""; drawTally(); });
  function drawTally() { const rows = $("tallyRows"); rows.textContent = ""; NAMES.forEach((n) => addRow(rows, [n, tally[n] || 0])); }

  const summary = { answers: 0, cls: {} };
  $("summaryReset").addEventListener("click", () => { summary.answers = 0; summary.cls = {}; $("summaryRows").textContent = ""; });

  function beep() {
    try { const ac = new (window.AudioContext || window.webkitAudioContext)(), o = ac.createOscillator(), g = ac.createGain();
      o.frequency.value = 880; g.gain.value = 0.15; o.connect(g); g.connect(ac.destination); o.start(); o.stop(ac.currentTime + 0.2); } catch (e) {}
  }
  function vibrate() { try { navigator.vibrate && navigator.vibrate(200); } catch (e) {} }
  function buzz() { vibrate(); beep(); }

  // ---------- 6. Extra credit: my-feature.js ----------
  const helper = {
    showMessage: (t) => { $("myMessage").textContent = t; show($("myMessage"), !!t); show($("myPanel"), true); },
    addToLog: (t) => { const li = document.createElement("li"); li.textContent = t; $("myLog").prepend(li); show($("myPanel"), true); },
    beep, vibrate,
  };
  function runMyFeature(res) {
    if (typeof myFeature !== "function") return;
    const r = { task: res.task, ms: res.ms, source: res.source };
    if (res.task === "classification") Object.assign(r, { answer: res.answer, confidence: res.confidence, scores: res.scores.map((s) => ({ ...s })) });
    else r.boxes = (res.shownBoxes || res.boxes).map((b) => ({ ...b }));
    try { myFeature(r, helper); show($("myError"), false); }
    catch (e) { $("myError").textContent = "Error in my-feature.js: " + e.message; show($("myError"), true); show($("myPanel"), true); }
  }

  // ---------- 7. Camera and photos ----------
  async function runOnce(src, w, h, kind) {
    const res = await predict(src, w, h, kind);
    lastResult = res; render(src, res);
    $("status").textContent = kind === "photo" ? "Photo" : "Live camera";
  }

  async function loop() {
    if (!video) return;
    if (!busy && video.readyState >= 2) {
      busy = true;
      try { await runOnce(video, video.videoWidth, video.videoHeight, "camera"); } finally { busy = false; }
    }
    requestAnimationFrame(loop);
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; video = null; $("camBtn").textContent = "Start camera";
  }

  $("camBtn").addEventListener("click", async () => {
    if (stream) { stopCamera(); $("status").textContent = "Camera off"; return; }
    show($("camErr"), false);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
      video = document.createElement("video"); video.playsInline = true; video.muted = true; video.srcObject = stream;
      await video.play(); $("camBtn").textContent = "Stop camera"; $("photoName").textContent = "Source: live camera";
      loop();
    } catch (e) {
      stopCamera();
      $("camErr").textContent = "The camera didn't start. Allow camera access for this page, or use “Test a saved photo” (on a phone it can take a photo too).";
      show($("camErr"), true);
    }
  });

  $("photo").addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0]; if (!f || !session) return;
    stopCamera();
    try {
      const bmp = await createImageBitmap(f, { imageOrientation: "from-image" });
      lastSource = bmp;
      $("photoName").textContent = "Photo: " + f.name;
      await runOnce(bmp, bmp.width, bmp.height, "photo");
    } catch (err) {
      $("camErr").textContent = "This photo couldn't be opened. Try a JPG or PNG."; show($("camErr"), true);
    }
    e.target.value = "";
  });

  // ---------- start ----------
  loadAll().then(() => {
    $("status").textContent = "Model ready";
    $("camBtn").disabled = false; $("photo").disabled = false; $("photoLabel").classList.remove("disabled");
    drawTally();
    ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, view.width, view.height);
  }).catch((e) => {
    $("status").textContent = "Model didn't load";
    setupError(e.message || String(e));
  });
})();

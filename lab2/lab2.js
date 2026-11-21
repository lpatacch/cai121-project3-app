// Lab 2: Edge AI on Your Phone. Uses the template's model folder (full model) and model-8bit.onnx in this folder.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const show = (el, on) => { el.hidden = !on; };
  const pct = (p) => `${Math.round(p * 100)}%`;
  const FILES = { full: "../model/model.onnx", small: "model-8bit.onnx" };
  const S = 224;
  let NAMES = [], session = null, version = "full", video = null, stream = null, busy = false, lastSrc = null;
  let times = [], firstMs = null, lastSpoken = "";
  const view = $("view"), ctx = view.getContext("2d");
  const work = document.createElement("canvas"); work.width = work.height = S;
  const wctx = work.getContext("2d", { willReadFrequently: true });

  function error(t) { $("err").textContent = t; show($("err"), !!t); }
  function netStatus() { $("net").textContent = navigator.onLine ? "on" : "off (airplane mode or no signal)"; }
  window.addEventListener("online", netStatus); window.addEventListener("offline", netStatus); netStatus();

  async function load(v) {
    $("camBtn").disabled = true; $("photo").disabled = true; $("photoLabel").classList.add("disabled");
    $("modelInfo").textContent = `Loading the ${v === "full" ? "full" : "compressed"} model…`; $("status").textContent = "Loading…";
    const t0 = performance.now();
    try {
      if (!NAMES.length) NAMES = await (await fetch("../model/classes.json")).json();
      const r = await fetch(FILES[v]);
      if (!r.ok) throw new Error(`${FILES[v]} is missing`);
      const bytes = new Uint8Array(await r.arrayBuffer());
      ort.env.wasm.wasmPaths = new URL("../ort/", location.href).href; ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      const secs = (performance.now() - t0) / 1000;
      $("modelInfo").textContent = `${v === "full" ? "Full (32-bit)" : "Compressed (8-bit)"}: ${(bytes.length / 1e6).toFixed(1)} MB, loaded in ${secs.toFixed(1)} s.`;
      times = []; firstMs = null; ["sFirst", "sMs", "sAvg"].forEach((id) => ($(id).textContent = "–"));
      $("status").textContent = "Model ready"; error("");
      $("camBtn").disabled = false; $("photo").disabled = false; $("photoLabel").classList.remove("disabled");
      if (lastSrc && !video) runOnce(lastSrc, lastSrc.width, lastSrc.height, "Photo");
    } catch (e) {
      $("status").textContent = "Model didn't load";
      error(navigator.onLine ? "The model didn't load: " + e.message : "The model didn't load because there's no internet. Load the page with internet first, then turn on airplane mode.");
    }
  }

  function prepare(src, w, h) {
    const r = Math.min(S / w, S / h), nw = Math.round(w * r), nh = Math.round(h * r);
    wctx.fillStyle = "#000"; wctx.fillRect(0, 0, S, S); wctx.imageSmoothingQuality = "high";
    wctx.drawImage(src, Math.floor((S - nw) / 2), Math.floor((S - nh) / 2), nw, nh);
    const px = wctx.getImageData(0, 0, S, S).data, t = new Float32Array(S * S * 3);
    for (let i = 0; i < S * S; i++) { t[3 * i] = px[4 * i]; t[3 * i + 1] = px[4 * i + 1]; t[3 * i + 2] = px[4 * i + 2]; }
    return new ort.Tensor("float32", t, [1, S, S, 3]);
  }

  async function runOnce(src, w, h, label) {
    const x = prepare(src, w, h), t0 = performance.now();
    const out = await session.run({ [session.inputNames[0]]: x });
    const ms = performance.now() - t0;
    if (firstMs === null) { firstMs = ms; $("sFirst").textContent = Math.round(ms); } else { times.push(ms); if (times.length > 30) times.shift(); }
    $("sMs").textContent = Math.round(ms);
    if (times.length) $("sAvg").textContent = Math.round(times.reduce((a, b) => a + b, 0) / times.length);
    const p = out[session.outputNames[0]].data;
    const top = Array.from(p, (c, i) => [i, c]).sort((a, b) => b[1] - a[1]).slice(0, 3);
    if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
    ctx.drawImage(src, 0, 0, w, h);
    const ns = +$("ns").value / 100, sure = top[0][1] >= ns;
    const text = sure ? NAMES[top[0][0]] : "Not sure";
    if (text !== lastSpoken) { $("answer").textContent = text; lastSpoken = text; }
    $("answerSub").textContent = sure ? `Confidence ${pct(top[0][1])}` : `Best guess ${NAMES[top[0][0]]} at ${pct(top[0][1])}, below ${pct(ns)}`;
    const tb = $("top3"); tb.textContent = "";
    top.forEach(([i, c], k) => { const tr = document.createElement("tr"); [k + 1, NAMES[i], pct(c)].forEach((v) => { const td = document.createElement("td"); td.textContent = v; tr.append(td); }); tb.append(tr); });
    $("status").textContent = label;
  }

  async function loop() {
    if (!video) return;
    if (!busy && video.readyState >= 2 && session) { busy = true; try { await runOnce(video, video.videoWidth, video.videoHeight, "Live camera"); } finally { busy = false; } }
    requestAnimationFrame(loop);
  }
  function stopCamera() { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; video = null; $("camBtn").textContent = "Start camera"; }

  $("camBtn").addEventListener("click", async () => {
    if (stream) { stopCamera(); $("status").textContent = "Camera off"; return; }
    show($("camErr"), false);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
      video = document.createElement("video"); video.playsInline = true; video.muted = true; video.srcObject = stream;
      await video.play(); $("camBtn").textContent = "Stop camera"; lastSrc = null; loop();
    } catch (e) { stopCamera(); $("camErr").textContent = "The camera didn't start. Allow camera access for this page, or use “Test a saved photo.”"; show($("camErr"), true); }
  });
  $("photo").addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0]; if (!f || !session) return;
    stopCamera();
    try { const bmp = await createImageBitmap(f, { imageOrientation: "from-image" }); lastSrc = bmp; await runOnce(bmp, bmp.width, bmp.height, "Photo"); }
    catch (err) { $("camErr").textContent = "This photo couldn't be opened. Try a JPG or PNG."; show($("camErr"), true); }
    e.target.value = "";
  });
  $("ns").addEventListener("input", () => { $("nsOut").textContent = $("ns").value + "%"; if (lastSrc && !video) runOnce(lastSrc, lastSrc.width, lastSrc.height, "Photo"); });
  for (const [id, v] of [["vFull", "full"], ["vSmall", "small"]]) {
    $(id).addEventListener("click", () => {
      if (version === v) return; version = v;
      $("vFull").setAttribute("aria-pressed", v === "full"); $("vSmall").setAttribute("aria-pressed", v === "small");
      load(v);
    });
  }
  load("full");
})();

/* SafeScan AI - módulo de detección de EPP en el navegador.
 * Requiere: onnxruntime-web (CDN), epp-logic.js, y los archivos
 *   models/ppe.onnx  (exportado con Ultralytics: format=onnx, imgsz=640)
 *   models/ppe.json  ({"names":[...], "imgsz":640})
 * Todo corre localmente: las imágenes no salen del dispositivo.
 */
(function () {
  const $ = s => document.querySelector(s);
  const L = window.EPPLogic;
  const CONF_THR = 0.25;
  const MODEL_URL = "models/ppe.onnx";
  const META_URL = "models/ppe.json";

  let session = null, meta = null, stream = null, running = false, busy = false;
  let lastLog = 0, fpsEMA = 0, lastDets = null, lastSize = null;
  const KEYS = ["helmet", "vest", "gloves", "boots", "goggles"];
  function getRequired() {
    const r = KEYS.filter(k => { const el = $("#eppReq_" + k); return el ? el.checked : true; });
    return r;
  }
  function evalNow(dets) { return L.evaluate(dets, { required: getRequired() }); }
  const work = document.createElement("canvas");
  const wctx = work.getContext("2d", { willReadFrequently: true });

  const COLORS = { ok: "#22c55e", mal: "#f97316", falta: "#ef4444", "n/d": "#93a0ae" };
  const LABEL = { ok: "Correcto", mal: "Mal colocado", falta: "Falta", "n/d": "Sin evidencia" };

  function setStatus(txt, live) {
    $("#eppStatus").textContent = txt;
    $("#eppDot").classList.toggle("live", !!live);
  }

  async function loadModel() {
    if (session) return true;
    if (!window.ort) { setStatus("No se pudo cargar onnxruntime-web (¿sin internet?)"); return false; }
    try {
      setStatus("Cargando modelo…");
      meta = await (await fetch(META_URL, { cache: "no-cache" })).json();
      session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
      $("#eppModelInfo").textContent = `Modelo: ${meta.names.length} clases · ${meta.imgsz}px`;
      setStatus("Modelo listo", false);
      return true;
    } catch (e) {
      console.error(e);
      setStatus("Modelo no encontrado en /models. Entrena y exporta uno (ver README).");
      return false;
    }
  }

  /* ---------- Preprocesado (letterbox) ---------- */
  function preprocess(src, sw, sh) {
    const S = meta.imgsz;
    work.width = S; work.height = S;
    const r = Math.min(S / sw, S / sh);
    const nw = Math.round(sw * r), nh = Math.round(sh * r);
    const dx = (S - nw) / 2, dy = (S - nh) / 2;
    wctx.fillStyle = "rgb(114,114,114)";
    wctx.fillRect(0, 0, S, S);
    wctx.drawImage(src, 0, 0, sw, sh, dx, dy, nw, nh);
    const { data } = wctx.getImageData(0, 0, S, S);
    const plane = S * S, f = new Float32Array(3 * plane);
    for (let i = 0, j = 0; i < plane; i++, j += 4) {
      f[i] = data[j] / 255; f[plane + i] = data[j + 1] / 255; f[2 * plane + i] = data[j + 2] / 255;
    }
    return { tensor: new ort.Tensor("float32", f, [1, 3, S, S]), r, dx, dy };
  }

  /* ---------- Postprocesado ---------- */
  function postprocess(out, pre, sw, sh) {
    const dims = out.dims, d = out.data, names = meta.names, nc = names.length;
    const dets = [];
    const push = (x1, y1, x2, y2, conf, ci) => {
      if (!names[ci]) return;
      dets.push({
        cls: names[ci], conf,
        x1: Math.max(0, (x1 - pre.dx) / pre.r), y1: Math.max(0, (y1 - pre.dy) / pre.r),
        x2: Math.min(sw, (x2 - pre.dx) / pre.r), y2: Math.min(sh, (y2 - pre.dy) / pre.r)
      });
    };
    if (dims.length === 3 && dims[2] === 6 && dims[1] !== 4 + nc) {
      // Modelo end-to-end (sin NMS): [1, N, 6] = x1,y1,x2,y2,conf,cls
      for (let i = 0; i < dims[1]; i++) {
        const o = i * 6, conf = d[o + 4];
        if (conf >= CONF_THR) push(d[o], d[o + 1], d[o + 2], d[o + 3], conf, Math.round(d[o + 5]));
      }
      return dets;
    }
    // Formato clásico: [1, 4+nc, N]
    const N = dims[2];
    for (let i = 0; i < N; i++) {
      let best = 0, bi = -1;
      for (let c = 0; c < nc; c++) { const s = d[(4 + c) * N + i]; if (s > best) { best = s; bi = c; } }
      if (best < CONF_THR) continue;
      const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
      push(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, bi);
    }
    return L.nms(dets, 0.5);
  }

  async function infer(src, sw, sh) {
    const pre = preprocess(src, sw, sh);
    const feeds = {}; feeds[session.inputNames[0]] = pre.tensor;
    const res = await session.run(feeds);
    return postprocess(res[session.outputNames[0]], pre, sw, sh);
  }

  /* ---------- Dibujo y resultados ---------- */
  function draw(dets, evals, sw, sh) {
    const cv = $("#eppOverlay");
    cv.width = sw; cv.height = sh;
    const g = cv.getContext("2d");
    g.clearRect(0, 0, sw, sh);
    g.lineWidth = Math.max(2, sw / 320);
    g.font = `600 ${Math.max(12, sw / 50)}px Archivo, sans-serif`;
    dets.forEach(d => {
      if (d.cls === "Person") return;
      const bad = d.cls.startsWith("no_");
      g.strokeStyle = bad ? COLORS.falta : "#3b82f6";
      g.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const t = `${d.cls} ${(d.conf * 100) | 0}%`;
      g.fillStyle = g.strokeStyle; g.fillRect(d.x1, d.y1 - 18, g.measureText(t).width + 8, 18);
      g.fillStyle = "#fff"; g.fillText(t, d.x1 + 4, d.y1 - 4);
    });
    evals.forEach((e, i) => {
      const p = e.person, col = e.compliant ? COLORS.ok : e.missing ? COLORS.falta : COLORS.mal;
      g.strokeStyle = col; g.lineWidth = Math.max(3, sw / 200);
      g.strokeRect(p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1);
      const t = `Persona ${i + 1}: ${e.compliant ? "CUMPLE" : "NO CUMPLE"}`;
      g.fillStyle = col; g.fillRect(p.x1, p.y2, g.measureText(t).width + 10, 20);
      g.fillStyle = "#000"; g.fillText(t, p.x1 + 5, p.y2 + 15);
    });
  }

  function chip(label, st) {
    return `<span class="chip" style="border-color:${COLORS[st]};color:${COLORS[st]}">${label}: ${LABEL[st]}</span>`;
  }

  function renderResults(evals) {
    const box = $("#eppResults");
    if (!evals.length) { box.innerHTML = `<div class="muted small">No se detectan personas en la imagen.</div>`; return; }
    box.innerHTML = evals.map((e, i) => `
      <div class="card" style="margin-bottom:8px;padding:12px">
        <div class="row between"><b>Persona ${i + 1}</b>
          <span class="pill ${e.compliant ? "ok" : "bad"}">${e.compliant ? "Cumple" : "No cumple"}</span></div>
        <div class="row" style="gap:6px;flex-wrap:wrap;margin-top:8px">${e.required.map(k => chip(L.ITEMS[k].label, e.items[k])).join("")}</div>
        ${e.issues.length ? `<ul class="small" style="margin:8px 0 0 18px">${e.issues.map(x => `<li>${x}</li>`).join("")}</ul>` : ""}
      </div>`).join("");
  }

  function maybeLog(evals) {
    const bad = evals.filter(e => e.missing || e.misplaced);
    if (!bad.length || Date.now() - lastLog < 20000 || typeof addHistory !== "function") return;
    lastLog = Date.now();
    const issues = [...new Set(bad.flatMap(e => e.issues))].join("; ");
    const a = (typeof asset === "function") ? asset() : { id: "-", name: "Zona monitoreada" };
    addHistory({
      type: "incident", code: "EPP-" + String(Date.now()).slice(-6), assetId: a.id, assetName: a.name,
      kind: "EPP", severity: bad.some(e => e.missing) ? "Alta" : "Media",
      desc: `${bad.length} persona(s) sin EPP correcto: ${issues}`, operator: "Detección automática", loto: false
    });
    if (typeof toast === "function") toast("Incumplimiento de EPP registrado en el historial");
  }

  /* ---------- Cámara / imagen ---------- */
  async function loop() {
    if (!running) return;
    const v = $("#eppVideo");
    if (!busy && v.videoWidth) {
      busy = true;
      const t0 = performance.now();
      try {
        const dets = await infer(v, v.videoWidth, v.videoHeight);
        const evals = evalNow(dets);
        draw(dets, evals, v.videoWidth, v.videoHeight);
        renderResults(evals); maybeLog(evals);
        const dt = performance.now() - t0; fpsEMA = fpsEMA ? fpsEMA * 0.8 + (1000 / dt) * 0.2 : 1000 / dt;
        setStatus(`Analizando · ${fpsEMA.toFixed(1)} FPS · ${dt.toFixed(0)} ms/frame`, true);
      } catch (e) { console.error(e); setStatus("Error en inferencia"); }
      busy = false;
    }
    requestAnimationFrame(loop);
  }

  async function startCam() {
    // 1) Cámara primero: así el permiso aparece de inmediato (el modelo tarda en cargar).
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) { setStatus("Este navegador no permite usar la cámara (se requiere https)"); return; }
    setStatus("Solicitando cámara…");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 960 } }, audio: false });
    } catch (e) {
      setStatus(e && e.name === "NotAllowedError" ? "Permiso de cámara denegado: actívalo en el navegador" : "No se pudo acceder a la cámara");
      return;
    }
    const v = $("#eppVideo"); v.srcObject = stream; await v.play();
    $("#eppPlaceholder").style.display = "none"; $("#eppStop").style.display = "";
    $("#eppSnap").style.display = ""; $("#eppAgain").style.display = "none";
    // 2) Modelo después (con aviso de progreso).
    setStatus("Cargando modelo de IA (~10 MB), espera unos segundos…");
    if (!(await loadModel())) return;
    running = true; loop();
  }

  function stopCam() {
    running = false;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    const v = $("#eppVideo"); if (v) v.srcObject = null;
    const s = $("#eppStop"); if (s) s.style.display = "none";
    const sn = $("#eppSnap"); if (sn) sn.style.display = "none";
    const p = $("#eppPlaceholder"); if (p) p.style.display = "";
    if (session) setStatus("Detenido");
  }
  window.eppStop = stopCam;


  async function snap() {
    const v = $("#eppVideo");
    if (!v.videoWidth) { setStatus("Espera a que la cámara muestre imagen"); return; }
    const w = v.videoWidth, h = v.videoHeight;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    c.getContext("2d").drawImage(v, 0, 0, w, h);
    running = false;
    while (busy) await new Promise(r => setTimeout(r, 50));
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    v.srcObject = null; v.style.display = "none";
    const still = $("#eppStill"); still.src = c.toDataURL("image/jpeg", 0.92); still.style.display = "block";
    $("#eppStop").style.display = "none"; $("#eppSnap").style.display = "none"; $("#eppAgain").style.display = "";
    setStatus("Analizando captura…");
    if (!(await loadModel())) return;
    const t0 = performance.now();
    const dets = await infer(c, w, h);
    const evals = evalNow(dets);
    lastDets = dets; lastSize = [w, h];
    draw(dets, evals, w, h); renderResults(evals); maybeLog(evals);
    setStatus(`Captura analizada · ${(performance.now() - t0).toFixed(0)} ms`);
  }

  function again() {
    $("#eppAgain").style.display = "none";
    $("#eppResults").innerHTML = "";
    const cv = $("#eppOverlay"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    $("#eppStill").style.display = "none"; $("#eppVideo").style.display = "";
    startCam();
  }

  async function fromFile(file) {
    if (!file || !(await loadModel())) return;
    stopCam();
    const img = new Image(); img.src = URL.createObjectURL(file);
    await img.decode();
    const v = $("#eppVideo"); v.style.display = "none";
    const still = $("#eppStill"); still.src = img.src; still.style.display = "block";
    $("#eppPlaceholder").style.display = "none";
    $("#eppStop").style.display = "none"; $("#eppSnap").style.display = "none"; $("#eppAgain").style.display = "";
    setStatus("Analizando imagen…");
    const t0 = performance.now();
    const dets = await infer(img, img.naturalWidth, img.naturalHeight);
    const evals = evalNow(dets);
    lastDets = dets; lastSize = [img.naturalWidth, img.naturalHeight];
    draw(dets, evals, img.naturalWidth, img.naturalHeight);
    renderResults(evals); maybeLog(evals);
    setStatus(`Imagen analizada · ${(performance.now() - t0).toFixed(0)} ms`);
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("#view-epp")) return;
    $("#eppCam").onclick = () => { $("#eppStill").style.display = "none"; $("#eppVideo").style.display = ""; startCam(); };
    $("#eppStop").onclick = snap;   // "Detener" congela la foto y la analiza
    $("#eppSnap").onclick = snap;
    $("#eppAgain").onclick = again;
    $("#eppFile").onchange = e => fromFile(e.target.files[0]);
    KEYS.forEach(k => {
      const el = $("#eppReq_" + k); if (!el) return;
      try { const sv = JSON.parse(localStorage.getItem("ss_epp_req") || "null"); if (sv) el.checked = sv.includes(k); } catch (e) {}
      el.onchange = () => {
        try { localStorage.setItem("ss_epp_req", JSON.stringify(getRequired())); } catch (e) {}
        if (lastDets && !running) { const ev = evalNow(lastDets); draw(lastDets, ev, lastSize[0], lastSize[1]); renderResults(ev); }
      };
    });
    const pre = () => { if (location.hash === "#epp") loadModel(); };
    window.addEventListener("hashchange", pre); pre();
  });
})();

/* SafeScan AI - lógica de cumplimiento de EPP (sin dependencias del DOM).
 * Entrada: detecciones de un modelo YOLO [{cls, conf, x1, y1, x2, y2}].
 * Salida: una evaluación por persona: estado de casco y chaleco
 *         ("ok" | "mal" | "falta" | "n/d").
 * Los umbrales geométricos (CFG) son heurísticas y deben calibrarse con el
 * conjunto de validación antes de reportarlos como resultados.
 */
(function (root) {
  const CFG = {
    personConf: 0.35,
    ppeConf: 0.30,
    // Región de cabeza: franja superior de la caja de la persona.
    headFrac: 0.30,
    // Región de torso: franja vertical de la caja de la persona.
    torsoTop: 0.15,
    torsoBottom: 0.60,
    // Casco: el centro del casco debe caer en la región de cabeza (con holgura).
    helmetCenterMaxFrac: 0.28,
    // Casco: fracción mínima de la caja del casco dentro de la caja de la persona.
    helmetInsideMin: 0.6,
    // Chaleco: fracción mínima del torso cubierta por el chaleco.
    vestTorsoCoverMin: 0.45,
    // Asociación de una caja a una persona: fracción de la caja dentro de la persona.
    assocMin: 0.5,
    // Botas: el centro debe estar en el 30 % inferior de la persona.
    bootsMinFrac: 0.70
  };

  const area = b => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  function inter(a, b) {
    const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
    const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
    return w > 0 && h > 0 ? w * h : 0;
  }
  const insideFrac = (box, container) => { const a = area(box); return a ? inter(box, container) / a : 0; };
  const center = b => ({ x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 });

  function iou(a, b) {
    const i = inter(a, b);
    const u = area(a) + area(b) - i;
    return u ? i / u : 0;
  }

  /** NMS por clase (para modelos no end-to-end). */
  function nms(dets, iouThr = 0.5) {
    const out = [];
    const byCls = {};
    dets.forEach(d => (byCls[d.cls] = byCls[d.cls] || []).push(d));
    Object.values(byCls).forEach(list => {
      list.sort((a, b) => b.conf - a.conf);
      const keep = [];
      list.forEach(d => { if (keep.every(k => iou(k, d) < iouThr)) keep.push(d); });
      out.push(...keep);
    });
    return out;
  }

  /** EPP soportados: clase "puesta" y clase "falta" del dataset. */
  const ITEMS = {
    helmet:  { ok: "helmet",  no: "no_helmet", label: "Casco" },
    vest:    { ok: "vest",    no: "no_vest",   label: "Chaleco" },
    gloves:  { ok: "gloves",  no: "no_gloves", label: "Guantes" },
    boots:   { ok: "boots",   no: "no_boots",  label: "Botas" },
    goggles: { ok: "goggles", no: "no_goggle", label: "Lentes" }
  };

  /**
   * @param dets  detecciones con nombre de clase en `cls`
   * @param cfg   sobrescribe CFG; `required` = lista de EPP exigidos (por defecto todos)
   * @returns [{person, items:{helmet,vest,gloves,boots,goggles}, required, helmet, vest,
   *            compliant, missing, misplaced, issues[]}]
   *          estado de cada ítem: "ok" | "mal" | "falta" | "n/d"
   */
  function evaluate(dets, cfg) {
    const C = Object.assign({}, CFG, cfg || {});
    const required = (cfg && cfg.required) || Object.keys(ITEMS);
    const persons = dets.filter(d => d.cls === "Person" && d.conf >= C.personConf);
    const ppe = dets.filter(d => d.cls !== "Person" && d.conf >= C.ppeConf);

    return persons.map(p => {
      const ph = p.y2 - p.y1;
      const torso = { x1: p.x1, x2: p.x2, y1: p.y1 + ph * C.torsoTop, y2: p.y1 + ph * C.torsoBottom };
      const head = { x1: p.x1, x2: p.x2, y1: p.y1, y2: p.y1 + ph * C.headFrac };
      const mine = ppe.filter(d => insideFrac(d, p) >= C.assocMin ||
        (d.cls.startsWith("no_") && insideFrac(d, p) >= 0.3));
      const of = (...names) => mine.filter(d => names.includes(d.cls));
      const cyFrac = d => (center(d).y - p.y1) / ph;

      const items = {}, issues = [];
      Object.keys(ITEMS).forEach(k => {
        const it = ITEMS[k], oks = of(it.ok), nos = of(it.no);
        let st = "n/d", msg = null;
        if (oks.length) {
          const b = oks.slice().sort((a, c) => c.conf - a.conf)[0];
          st = "ok";
          if (k === "helmet" && (cyFrac(b) > C.helmetCenterMaxFrac || insideFrac(b, p) < C.helmetInsideMin)) {
            st = "mal"; msg = "Casco presente pero fuera de la cabeza";
          } else if (k === "vest" && inter(b, torso) / area(torso) < C.vestTorsoCoverMin) {
            st = "mal"; msg = "Chaleco mal colocado o cubre poco el torso";
          } else if (k === "goggles" && cyFrac(b) > C.headFrac) {
            st = "mal"; msg = "Lentes fuera de la zona de los ojos";
          } else if (k === "boots" && cyFrac(b) < C.bootsMinFrac) {
            st = "mal"; msg = "Botas fuera de la zona de los pies";
          } else if ((k === "gloves" || k === "boots") && nos.length) {
            st = "mal"; msg = it.label + ": solo uno puesto (falta el otro)";
          }
        } else if (nos.length) {
          st = "falta"; msg = "Sin " + it.label.toLowerCase();
        }
        items[k] = st;
        if (msg && required.includes(k)) issues.push(msg);
      });

      const reqStates = required.map(k => items[k]);
      const compliant = reqStates.length > 0 && reqStates.every(x => x === "ok");
      return {
        person: p, head, torso, items, required,
        helmet: items.helmet, vest: items.vest,
        compliant, missing: reqStates.includes("falta"), misplaced: reqStates.includes("mal"), issues
      };
    });
  }

  const api = { CFG, ITEMS, evaluate, nms, iou, insideFrac };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EPPLogic = api;
})(typeof self !== "undefined" ? self : this);

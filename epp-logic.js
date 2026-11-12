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
    assocMin: 0.5
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

  /**
   * @param dets  detecciones con nombre de clase en `cls`
   * @param cfg   sobrescribe CFG si se requiere
   * @returns [{person, helmet, vest, compliant, issues[]}]
   */
  function evaluate(dets, cfg) {
    const C = Object.assign({}, CFG, cfg || {});
    const persons = dets.filter(d => d.cls === "Person" && d.conf >= C.personConf);
    const ppe = dets.filter(d => d.cls !== "Person" && d.conf >= C.ppeConf);

    return persons.map(p => {
      const ph = p.y2 - p.y1;
      const head = { x1: p.x1, x2: p.x2, y1: p.y1, y2: p.y1 + ph * C.headFrac };
      const torso = { x1: p.x1, x2: p.x2, y1: p.y1 + ph * C.torsoTop, y2: p.y1 + ph * C.torsoBottom };
      const mine = ppe.filter(d => insideFrac(d, p) >= C.assocMin ||
        (d.cls.startsWith("no_") && insideFrac(d, p) >= 0.3));
      const of = (...names) => mine.filter(d => names.includes(d.cls));

      const issues = [];
      let helmet = "n/d", vest = "n/d";

      // --- Casco ---
      const hOk = of("helmet"), hNo = of("no_helmet");
      if (hOk.length) {
        const h = hOk.sort((a, b) => b.conf - a.conf)[0];
        const cy = (center(h).y - p.y1) / ph;
        const misplaced = cy > C.helmetCenterMaxFrac || insideFrac(h, p) < C.helmetInsideMin;
        helmet = misplaced ? "mal" : "ok";
        if (misplaced) issues.push("Casco presente pero fuera de la cabeza");
      } else if (hNo.length) {
        helmet = "falta"; issues.push("Sin casco");
      }

      // --- Chaleco ---
      const vOk = of("vest"), vNo = of("no_vest");
      if (vOk.length) {
        const v = vOk.sort((a, b) => b.conf - a.conf)[0];
        const cover = inter(v, torso) / area(torso);
        vest = cover < C.vestTorsoCoverMin ? "mal" : "ok";
        if (vest === "mal") issues.push("Chaleco mal colocado o cubre poco el torso");
      } else if (vNo.length) {
        vest = "falta"; issues.push("Sin chaleco");
      }

      const compliant = helmet === "ok" && vest === "ok";
      return { person: p, head, torso, helmet, vest, compliant, issues };
    });
  }

  const api = { CFG, evaluate, nms, iou, insideFrac };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EPPLogic = api;
})(typeof self !== "undefined" ? self : this);

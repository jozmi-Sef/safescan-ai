const assert = require("assert");
const { evaluate } = require("../epp-logic.js");
const P = { cls: "Person", conf: .9, x1: 100, y1: 100, x2: 200, y2: 400 }; // alto 300
const box = (cls, x1, y1, x2, y2, conf = .9) => ({ cls, conf, x1, y1, x2, y2 });
const run2 = d => evaluate([P, ...d], { required: ["helmet", "vest"] })[0]; // pruebas de casco/chaleco
const run = d => evaluate([P, ...d])[0];                                  // todos los EPP requeridos

// Casco bien puesto + chaleco cubriendo el torso -> cumple
let r = run2([box("helmet", 120, 95, 180, 150), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "ok"); assert.equal(r.vest, "ok"); assert(r.compliant);

// Sin casco (no_helmet) -> falta
r = run2([box("no_helmet", 120, 100, 180, 150), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "falta"); assert(!r.compliant);

// Casco en la mano (centro lejos de la cabeza) -> mal colocado
r = run2([box("helmet", 170, 250, 200, 290), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "mal");

// Chaleco que cubre poco torso (abierto/parcial) -> mal colocado
r = run2([box("helmet", 120, 95, 180, 150), box("vest", 100, 150, 120, 290)]);
assert.equal(r.vest, "mal");

// Sin chaleco -> falta
r = run2([box("helmet", 120, 95, 180, 150), box("no_vest", 105, 150, 195, 290)]);
assert.equal(r.vest, "falta");

// Sin evidencia -> n/d y no cumple
r = run2([]); assert.equal(r.helmet, "n/d"); assert(!r.compliant);

// EPP de otra persona no se asigna
r = run2([box("helmet", 500, 95, 560, 150)]); assert.equal(r.helmet, "n/d");

// --- Nuevos EPP: guantes, botas, lentes ---
const full = [box("helmet",120,95,180,150), box("vest",105,150,195,290), box("goggles",130,110,170,125),
              box("gloves",100,260,115,285), box("boots",110,370,190,400)];
r = run(full); assert(r.compliant, "todo puesto debe cumplir"); assert.equal(r.items.goggles,"ok"); assert.equal(r.items.boots,"ok");
// Sin guantes -> no cumple (si guantes es requerido)
r = run([...full.filter(d=>d.cls!=="gloves"), box("no_gloves",100,260,115,285)]);
assert.equal(r.items.gloves,"falta"); assert(!r.compliant); assert(r.missing);
// ...pero si guantes NO es requerido, sí cumple
r = evaluate([P, ...full.filter(d=>d.cls!=="gloves"), box("no_gloves",100,260,115,285)], {required:["helmet","vest","goggles","boots"]})[0];
assert(r.compliant);
// Un solo guante puesto -> parcial = mal
r = run([...full, box("no_gloves",185,260,198,285)]); assert.equal(r.items.gloves,"mal");
// Botas arriba (en la mano) -> mal
r = run([...full.filter(d=>d.cls!=="boots"), box("boots",100,200,190,240)]); assert.equal(r.items.boots,"mal");
// Lentes en la frente/pecho -> mal
r = run([...full.filter(d=>d.cls!=="goggles"), box("goggles",130,200,170,215)]); assert.equal(r.items.goggles,"mal");
console.log("OK: pruebas de lógica de EPP pasaron");

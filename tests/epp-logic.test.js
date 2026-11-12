const assert = require("assert");
const { evaluate } = require("../epp-logic.js");
const P = { cls: "Person", conf: .9, x1: 100, y1: 100, x2: 200, y2: 400 }; // alto 300
const box = (cls, x1, y1, x2, y2, conf = .9) => ({ cls, conf, x1, y1, x2, y2 });
const run = d => evaluate([P, ...d])[0];

// Casco bien puesto + chaleco cubriendo el torso -> cumple
let r = run([box("helmet", 120, 95, 180, 150), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "ok"); assert.equal(r.vest, "ok"); assert(r.compliant);

// Sin casco (no_helmet) -> falta
r = run([box("no_helmet", 120, 100, 180, 150), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "falta"); assert(!r.compliant);

// Casco en la mano (centro lejos de la cabeza) -> mal colocado
r = run([box("helmet", 170, 250, 200, 290), box("vest", 105, 150, 195, 290)]);
assert.equal(r.helmet, "mal");

// Chaleco que cubre poco torso (abierto/parcial) -> mal colocado
r = run([box("helmet", 120, 95, 180, 150), box("vest", 100, 150, 120, 290)]);
assert.equal(r.vest, "mal");

// Sin chaleco -> falta
r = run([box("helmet", 120, 95, 180, 150), box("no_vest", 105, 150, 195, 290)]);
assert.equal(r.vest, "falta");

// Sin evidencia -> n/d y no cumple
r = run([]); assert.equal(r.helmet, "n/d"); assert(!r.compliant);

// EPP de otra persona no se asigna
r = run([box("helmet", 500, 95, 560, 150)]); assert.equal(r.helmet, "n/d");
console.log("OK: 7 pruebas de lógica de EPP pasaron");

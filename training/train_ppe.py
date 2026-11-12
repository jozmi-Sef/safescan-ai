"""Entrena y exporta el detector de EPP para SafeScan AI.

Ejecutar en Google Colab (GPU T4) o local con GPU:
    pip install ultralytics
    python train_ppe.py            # ~30-60 min en T4

Genera:
    runs/ppe/weights/best.pt
    ../models/ppe.onnx  y  ../models/ppe.json   (los usa la web)
    ../models/metrics.json                      (para el informe, PPT y póster)
    runs/ppe/*.png                              (matriz de confusión, curvas PR, etc.)

Dataset: Construction-PPE de Ultralytics (se descarga solo). Cítalo en el informe
(ver README). Los nombres de clase se leen del dataset; no se asumen.
"""
import json, shutil
from pathlib import Path
from ultralytics import YOLO

EPOCHS, IMGSZ, BATCH, SEED = 100, 640, 16, 0
BASE = "yolo11n.pt"          # modelo pequeño: corre en el navegador
OUT = Path(__file__).resolve().parent.parent / "models"
OUT.mkdir(exist_ok=True)

model = YOLO(BASE)
model.train(data="construction-ppe.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            seed=SEED, project="runs", name="ppe", exist_ok=True)

best = YOLO("runs/ppe/weights/best.pt")

# Métricas reales sobre el conjunto de TEST (no visto durante el entrenamiento)
m = best.val(data="construction-ppe.yaml", split="test", imgsz=IMGSZ, plots=True)
names = best.names
metrics = {
    "base_model": BASE, "epochs": EPOCHS, "imgsz": IMGSZ, "seed": SEED,
    "split": "test",
    "precision": float(m.box.mp), "recall": float(m.box.mr),
    "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
    "per_class": {names[int(c)]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                  "AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])}
                  for i, c in enumerate(m.box.ap_class_index)},
    "speed_ms": {k: float(v) for k, v in m.speed.items()},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

# Exportación para el navegador (onnxruntime-web)
onnx_path = best.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False)
shutil.copy(onnx_path, OUT / "ppe.onnx")
(OUT / "ppe.json").write_text(json.dumps(
    {"names": [names[i] for i in range(len(names))], "imgsz": IMGSZ}, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))

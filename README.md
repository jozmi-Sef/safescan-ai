# SafeScan AI — Detección de EPP

App web (HTML + JS) que combina el escáner de activos con un detector de EPP
(casco, chaleco, guantes, botas, lentes) que corre 100 % en el navegador con YOLO + onnxruntime-web.

## Cómo funciona la detección
1. YOLO detecta `Person`, `helmet`, `vest`, `no_helmet`, … (dataset Construction-PPE).
2. `epp-logic.js` asocia cada EPP a una persona y evalúa **geometría**:
   - **Casco mal colocado**: hay casco, pero su centro no está en la zona de la cabeza.
   - **Chaleco mal colocado**: hay chaleco, pero cubre menos del 45 % del torso.
   - **Falta**: el modelo detecta `no_helmet` / `no_vest`.
3. Los incumplimientos se guardan en el Historial.

## Entrenar el modelo (obligatorio: el repo no incluye pesos)
1. Abre Google Colab → GPU T4.
2. `pip install ultralytics` y sube `training/train_ppe.py`.
3. `python train_ppe.py`
4. Descarga `models/ppe.onnx`, `models/ppe.json` y `models/metrics.json` a la carpeta `models/` del repo.
5. Publica en GitHub Pages. Sin `ppe.onnx` la pestaña EPP avisa que falta el modelo.

## Pruebas de la lógica
`node tests/epp-logic.test.js`

## Declaración de uso de IA
El código de la aplicación fue desarrollado con asistencia de Claude (Anthropic).
Debe declararse en la Introducción del informe y en el PPT (rúbrica del curso).

## Dataset
Ultralytics Construction-PPE. Revisa su licencia y cita según su documentación:
https://docs.ultralytics.com/datasets/detect/construction-ppe/

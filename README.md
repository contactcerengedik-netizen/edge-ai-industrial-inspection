# Bottle Cap Defect Detection

Real-time edge AI inspection: train YOLO11n, export ONNX, run on Raspberry Pi 5 + IMX500 camera, stabilize labels, publish over MQTT.

Python · YOLO11 · Ultralytics · OpenCV · ONNX · Raspberry Pi 5 · MQTT · Docker

## Results

| Metric | Value |
|---|---|
| Precision | 82.6% |
| Recall | 82.9% |
| mAP50 | 81.6% |
| Tesla T4 inference | 13.8 ms |
| Raspberry Pi 5 (ONNX, CPU) | ~137–150 ms · ~6.7–7.3 FPS |

<img src="results/predictions/104_jpg.rf.7e2601bec1f456567de13bbfd1c1b505.jpg" width="240">
<img src="results/predictions/107_jpg.rf.ce2c6f0af1626e183e9c3b6e39abf13f.jpg" width="240">
<img src="results/predictions/113_jpg.rf.ebb47483cdb619c748309dd9e11490e6.jpg" width="240">

## Pipeline

Camera → YOLO11n (ONNX) → stable label → MQTT `inspection/result`

## Data

Public dataset: Roboflow Universe, bottle-cap-defect-2, version 8.
1435 training images, 107 validation images.
The images were not collected by me.

## Training

40 epochs, image size 640, batch 16, pretrained YOLO11n, Tesla T4.

![training curves](results/results.png)
![confusion matrix](results/confusion_matrix.png)

## Edge deployment

- Exported YOLO11n to ONNX; ran inference on Raspberry Pi 5 with onnxruntime
- Live camera path uses Picamera2 + IMX500 AI Camera
- Temporal stability filter (`STABLE_N=3`) before publishing labels
- MQTT publish on topic `inspection/result`
- Mosquitto broker via Docker Compose

Scripts:
- [`inference/benchmark_onnx.py`](inference/benchmark_onnx.py)
- [`inference/live_onnx.py`](inference/live_onnx.py)
- [`docker-compose.yml`](docker-compose.yml)

```bash
# broker
sudo systemctl stop mosquitto   # if apt mosquitto is running
docker compose up -d
mosquitto_sub -h localhost -t inspection/result -v

# live camera (host; needs Picamera2)
source .venv/bin/activate
python inference/live_onnx.py
```

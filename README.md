# Bottle Cap Defect Detection

Fine-tuned YOLO11n to detect bottle-cap condition from a public dataset.

Python · YOLO11 · Ultralytics · OpenCV · ONNX · Raspberry Pi 5

**Precision 82.6% · Recall 82.9% · mAP50 81.6%**

| Device | Latency | FPS |
|---|---|---|
| Tesla T4 | 13.8 ms | — |
| Raspberry Pi 5 (ONNX, CPU) | 136.9 ms | 7.3 |

<img src="results/predictions/104_jpg.rf.7e2601bec1f456567de13bbfd1c1b505.jpg" width="240">
<img src="results/predictions/107_jpg.rf.ce2c6f0af1626e183e9c3b6e39abf13f.jpg" width="240">
<img src="results/predictions/113_jpg.rf.ebb47483cdb619c748309dd9e11490e6.jpg" width="240">

## Pipeline

Image → YOLO11n → defect / good / loos-cap / no-cap / ring-missing

## Data

Public dataset: Roboflow Universe, bottle-cap-defect-2, version 8.
1435 training images, 107 validation images.
The images were not collected by me.

## Training

40 epochs, image size 640, batch 16, pretrained YOLO11n, Tesla T4.

![training curves](results/results.png)
![confusion matrix](results/confusion_matrix.png)

## Edge deployment

Exported YOLO11n to ONNX and ran inference on Raspberry Pi 5 with onnxruntime.
Benchmark: 50 runs after 5 warm-up passes, image size 640.

Script: [`inference/benchmark_onnx.py`](inference/benchmark_onnx.py)

#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "bottle-cap-yolo11n.onnx"
IMAGE = ROOT / "test.jpg"
OUT = ROOT / "inference" / "out.jpg"
NAMES = ["defect", "good", "loos-cap", "no-cap", "ring-missing"]
SIZE = 640
CONF = 0.4
IOU = 0.45
WARMUP = 5
RUNS = 50


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, left, top


def preprocess(im):
    img, r, left, top = letterbox(im, SIZE)
    x = img[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], r, left, top


def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_thres):
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF, iou_thres)
    if len(idxs) == 0:
        return []
    return np.array(idxs).reshape(-1)


def postprocess(out, r, left, top, orig_shape):
    pred = np.squeeze(out).T
    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(1)
    scores = scores_all.max(1)
    keep = scores >= CONF
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if len(boxes) == 0:
        return []
    boxes = xywh2xyxy(boxes)
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, :4] /= r
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    wh = boxes.copy()
    wh[:, 2] = boxes[:, 2] - boxes[:, 0]
    wh[:, 3] = boxes[:, 3] - boxes[:, 1]
    idxs = nms(wh, scores, IOU)
    return [(boxes[i], float(scores[i]), int(cls_ids[i])) for i in idxs]


def main():
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    im0 = cv2.imread(str(IMAGE))
    x, r, left, top = preprocess(im0)

    for _ in range(WARMUP):
        session.run(None, {inp_name: x})

    times = []
    out = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = session.run(None, {inp_name: x})[0]
        times.append((time.perf_counter() - t0) * 1000)

    dets = postprocess(out, r, left, top, im0.shape)
    vis = im0.copy()
    for box, score, cid in dets:
        x1, y1, x2, y2 = box.astype(int)
        label = f"{NAMES[cid]} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(OUT), vis)

    mean = sum(times) / len(times)
    print(f"detections: {len(dets)}")
    for box, score, cid in dets:
        print(f"  {NAMES[cid]} {score:.3f}")
    print(f"latency_ms: {mean:.1f}")
    print(f"fps: {1000 / mean:.1f}")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

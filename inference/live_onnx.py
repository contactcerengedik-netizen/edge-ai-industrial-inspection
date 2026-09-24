#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "bottle-cap-yolo11n.onnx"
OUT = ROOT / "inference" / "live.jpg"
NAMES = ["defect", "good", "loos-cap", "no-cap", "ring-missing"]
SIZE = 640
CONF = 0.4
IOU = 0.45
STABLE_N = 3


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

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    print("Canli inference basladi. Durdurmak icin Ctrl+C")
    frame_id = 0
    pending_label = None
    pending_count = 0
    stable_label = "none"

    try:
        while True:
            rgb = picam2.capture_array()
            im0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x, r, left, top = preprocess(im0)

            t0 = time.perf_counter()
            out = session.run(None, {inp_name: x})[0]
            ms = (time.perf_counter() - t0) * 1000
            fps = 1000 / ms if ms > 0 else 0

            dets = postprocess(out, r, left, top, im0.shape)
            labels = [f"{NAMES[c]} {s:.2f}" for _, s, c in dets] or ["none"]

            current = labels[0].split()[0] if dets else "none"
            if current == pending_label:
                pending_count += 1
            else:
                pending_label = current
                pending_count = 1
            if pending_count >= STABLE_N:
                stable_label = current

            print(
                f"[{frame_id}] {ms:.0f} ms | {fps:.1f} FPS | raw={', '.join(labels)} | stable={stable_label}"
            )

            vis = im0.copy()
            for box, score, cid in dets:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{NAMES[cid]} {score:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                vis,
                f"{fps:.1f} FPS | {stable_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imwrite(str(OUT), vis)
            frame_id += 1
    except KeyboardInterrupt:
        print("\nDurdu. Son kare:", OUT)
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()

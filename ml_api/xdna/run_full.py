"""YoloNpu end to end: per-frame time and detections against fp32 ONNX Runtime."""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/w")
from obico_post import post_processing
from yolo_npu import YoloNpu

MODEL = "/w/model.onnx"
net = YoloNpu(MODEL, json.load(open("/w/scales.json")), "/cache/rt")
print("programs:", len(net.programs), sorted(net.programs))
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def prep(path):
    img = cv2.imread(path)
    x = cv2.cvtColor(cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return np.expand_dims(np.transpose(x, (2, 0, 1)).astype(np.float32), 0) / 255.0, img.shape


def dets(out, shape):
    return post_processing(out, shape[1], shape[0], 0.08, 0.45, ["failure"])[0]


paths = [f"/w/{n}.jpg" for n in ("detached-1", "gaps-1", "bubbling-1")] + sorted(glob.glob("/w/calib/real/*.jp*g")) + sorted(glob.glob("/w/calib/obico/*.jpg"))
x, _ = prep(paths[0])
net.run(x)                                   # warm-up
walls, cpus, drifts, mism = [], [], [], 0
for p in paths:
    x, shape = prep(p)
    t, c = time.perf_counter(), time.process_time()
    out = net.run(x)
    walls.append(time.perf_counter() - t)
    cpus.append(time.process_time() - c)
    ref = sess.run(None, {"input": x})
    dr, dq = dets(ref, shape), dets(out, shape)
    tr = max([d[1] for d in dr], default=0.0)
    tq = max([d[1] for d in dq], default=0.0)
    drifts.append(abs(tr - tq))
    mism += len(dr) != len(dq)
    if p in paths[:3] or len(dr) != len(dq) or abs(tr - tq) > 0.02:
        print(f"  {os.path.basename(p)[:38]:38} fp32 {len(dr)} top {tr:.3f} | npu {len(dq)} top {tq:.3f}")
print(f"{len(paths)} frames: wall median {np.median(walls)*1e3:.0f} ms (min {min(walls)*1e3:.0f}), "
      f"CPU median {np.median(cpus)*1e3:.0f} ms | top-detection drift max {max(drifts):.3f} mean {np.mean(drifts):.4f}, "
      f"detection-count mismatches {mism}/{len(paths)}")

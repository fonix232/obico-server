"""int8 scheme for the NPU, calibrated and checked on the CPU.

Weights: symmetric int8 per output channel. Activations: symmetric int8 per
conv input, scale from calibration. Products accumulate exactly (int32 on the
NPU, float64 here). Bias, leaky and the requantisation happen on the
accumulator. Detections are compared with the fp32 model through Obico's own
post-processing, on images left out of calibration.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/w")
from yolo_host import YoloV2
from obico_post import post_processing

THRESH, NMS = 0.08, 0.45
net = YoloV2("/w/model.onnx")
names = ["failure"]


def prep(path):
    img = cv2.imread(path)
    x = cv2.cvtColor(cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return np.expand_dims(np.transpose(x, (2, 0, 1)).astype(np.float32), 0) / 255.0, img.shape


real = sorted(glob.glob("/w/calib/real/*.jp*g"))
obico = sorted(glob.glob("/w/calib/obico/*.jpg"))   # photos only; the PNGs are UI screenshots
calib = real[::3] + real[1::3] + obico[::2]          # ~2/3 of the real frames, half of Obico's
held = [p for p in real + obico if p not in calib] + [f"/w/{n}.jpg" for n in ("detached-1", "gaps-1", "bubbling-1")]
print(f"calibration {len(calib)} images, held out {len(held)}")

# --- calibration: a sample of every conv input
rng = np.random.default_rng(0)
samples = {l.index: [] for l in net.convs}


def record(layer, a):
    flat = a.reshape(-1)
    samples[layer.index].append(flat[rng.integers(0, flat.size, 100_000)])
    return a @ layer.weight


for p in calib:
    net.run(prep(p)[0], record)


def qparams(lo, hi):
    """Asymmetric int8: real = s * (q - zp), q in [-128, 127]."""
    s = max(hi - lo, 1e-12) / 255.0
    zp = int(np.clip(np.round(-128 - lo / s), -128, 127))
    return s, zp


def fake_quant(x, s, zp):
    return s * (np.clip(np.round(x / s) + zp, -128, 127) - zp)


mode = os.environ.get("CLIP", "mse")
s_in, zp_in = {}, {}
for i, chunks in samples.items():
    v = np.concatenate(chunks).astype(np.float64)
    lo0, hi0 = float(v.min()), float(v.max())
    if mode == "mse":
        best = None
        for fh in np.linspace(0.2, 1.0, 33):          # shrink the positive side
            for fl in (1.0, 0.75, 0.5, 0.25):          # and the (small) negative side
                s, zp = qparams(min(lo0 * fl, 0.0), hi0 * fh)
                err = np.mean((fake_quant(v, s, zp) - v) ** 2)
                if best is None or err < best[0]:
                    best = (err, s, zp)
        _, s_in[i], zp_in[i] = best
    else:
        s_in[i], zp_in[i] = qparams(min(lo0, 0.0), hi0)

s_w = {l.index: np.maximum(np.abs(l.weight).max(axis=0), 1e-12) / 127.0 for l in net.convs}
w_q = {l.index: np.clip(np.round(l.weight / s_w[l.index]), -127, 127) for l in net.convs}
json.dump({"clip": mode, "s_in": s_in, "zp_in": zp_in}, open(f"/w/scales_{mode}.json", "w"), indent=1)


def int8_matmul(layer, a):
    i = layer.index
    x_q = np.clip(np.round(a / s_in[i]) + zp_in[i], -128, 127)
    acc = x_q.astype(np.float64) @ w_q[i].astype(np.float64)      # what the NPU computes (int32)
    acc -= zp_in[i] * w_q[i].sum(axis=0)                           # zero point, folded into the bias
    return (acc * (s_in[i] * s_w[i])).astype(np.float32)


def detections(out, shape):
    h, w = shape[:2]
    return post_processing(out, w, h, THRESH, NMS, names)[0]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def evaluate(matmul, label, verbose=False):
    top_diffs, mismatches, min_iou = [], 0, 1.0
    for p in held:
        x, shape = refs[p][1], refs[p][2]
        ref, dr = refs[p][0], refs[p][3]
        q = net.run(x, matmul)
        dq = detections(q, shape)
        top_r = max([d[1] for d in dr], default=0.0)
        top_q = max([d[1] for d in dq], default=0.0)
        top_diffs.append(abs(top_r - top_q))
        mismatches += len(dr) != len(dq)
        for d in dr:
            if d[1] > 0.2:
                min_iou = min(min_iou, max((iou(d[2], e[2]) for e in dq), default=0.0))
        if verbose:
            print(f"  {os.path.basename(p)[:40]:40} fp32 {len(dr)} top {top_r:.3f} | int8 {len(dq)} top {top_q:.3f}")
    print(f"{label}: top-detection drift max {max(top_diffs):.3f} mean {np.mean(top_diffs):.3f}, count mismatches {mismatches}/{len(held)}, min IoU {min_iou:.3f}", flush=True)
    return max(top_diffs)


refs = {}
for p in held:
    x, shape = prep(p)
    ref = net.run(x, lambda layer, a: a @ layer.weight)
    refs[p] = (ref, x, shape, detections(ref, shape))

evaluate(int8_matmul, f"all int8 ({mode})", verbose=True)
if os.environ.get("MIXED"):
    def bf16(a):
        import ml_dtypes
        return a.astype(ml_dtypes.bfloat16).astype(np.float32)
    for first_int8 in (6, 8, 10, 13):
        macs = sum(refs[held[0]][1].shape[0] for _ in [0])  # placeholder to keep flake quiet
        evaluate(lambda layer, a, f=first_int8: int8_matmul(layer, a) if f <= layer.index < 22
                 else bf16(a) @ bf16(layer.weight), f"bf16 conv00-{first_int8-1:02d}, int8 conv{first_int8:02d}-21")
if os.environ.get("SENSITIVITY"):
    for j in [l.index for l in net.convs if l.index != 22]:
        evaluate(lambda layer, a, j=j: int8_matmul(layer, a) if layer.index == j else a @ layer.weight, f"only conv{j:02d} int8")

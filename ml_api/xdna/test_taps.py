"""Milestone 3 check: implicit im2col (DMA taps) + output into the next padded buffer.

conv00 (3 channels stored as 4), conv02 (3x3, C=64), conv03 (1x1, C=128) with
real activations; the valid output region is compared with a host reference.
"""
import importlib.util
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/w")
import aie.iron as iron
from aie.iron.device import from_name
import npu_xrt
from pack import BF16, b_src_rowmaj, epilogue_params_bf16, param_chunk
from yolo_host import YoloV2

spec = importlib.util.spec_from_file_location("conv_rt", "/w/conv_rt.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
iron.set_current_device(from_name("npu", n_cols=4))
net = YoloV2("/w/model.onnx")
onnx_w = {}
import onnx
from onnx import numpy_helper
mdl = onnx.load("/w/model.onnx")
inits = {i.name: numpy_helper.to_array(i) for i in mdl.graph.initializer}
convs = [n for n in mdl.graph.node if n.op_type == "Conv"]

img = cv2.imread("/w/detached-1.jpg")
x = cv2.cvtColor(cv2.resize(img, (416, 416)), cv2.COLOR_BGR2RGB)
x = np.expand_dims(np.transpose(x, (2, 0, 1)).astype(np.float32), 0) / 255.0
acts = {}  # conv index -> its input activation [H, W, C] (float32)
net_in = {}


def capture(layer, a):
    net_in[layer.index] = a
    return a @ layer.weight


net.run(x, capture)


def ceil(v, a):
    return -(-v // a) * a


def conv_input_hwc(idx, h, w):
    a = net_in[idx]
    if net.convs[idx].k == 1:
        return a.reshape(h, w, -1)
    # im2col rows hold (C, kh, kw); the centre tap is the pixel itself
    c = a.shape[1] // 9
    return a.reshape(h, w, c, 3, 3)[:, :, :, 1, 1]


def padded(act, w2, rows, c_store):
    h, w, c = act.shape
    buf = np.zeros((rows, c_store), BF16)
    buf[:(h + 2) * w2].reshape(h + 2, w2, c_store)[1:h + 1, 1:w + 1, :c] = act
    return buf


tests = [(0, 416, 416, (64, 32, 16)), (2, 104, 104, (64, 64, 32)), (3, 104, 104, (64, 32, 16))]
cases = []
for idx, h, w, (m, k, n) in tests:
    wgt = inits[convs[idx].input[1]]          # [Cout, Cin, kh, kw]
    bias = inits[convs[idx].input[2]]
    cout, cin, kh, _ = wgt.shape
    w2 = w + 2
    c_store = 4 if cin == 3 else cin
    M = ceil(h * w2, 8 * m)
    if kh == 3 and c_store >= k:
        K = 9 * c_store
        Wk = wgt.transpose(2, 3, 1, 0).reshape(K, cout)                  # (dy, dx, c)
    elif kh == 3:
        K = 3 * k
        Wk = np.zeros((K, cout), np.float32)
        for dy in range(3):
            for dx in range(3):
                Wk[dy * k + dx * c_store: dy * k + dx * c_store + cin] = wgt[:, :, dy, dx].T
    else:
        K = cin
        Wk = wgt[:, :, 0, 0].T
    N = ceil(cout, n * 4)
    a_rows = M + 2 * w2 + 2 + k
    c_shift = w2 + 1
    c_rows = max(c_shift + M, (h + 2) * w2)
    act = conv_input_hwc(idx, h, w).astype(BF16)
    A = padded(act.astype(np.float32), w2, a_rows, c_store)
    B = np.zeros((k + K, N), BF16)
    B[k:, :cout] = Wk
    src = b_src_rowmaj(k, n, 8, 4)
    for j in range(N // n):
        bj = np.zeros(n, np.float32)
        valid = max(0, min(n, cout - j * n))
        bj[:valid] = bias[j * n:j * n + valid]
        B[:k, j * n:(j + 1) * n] = param_chunk(epilogue_params_bf16(bj), src, (k, n), BF16)
    # reference: true conv on the bf16 input, bias, leaky, bf16
    ref = net_in[idx].astype(BF16).astype(np.float32) @ net.convs[idx].weight.astype(BF16).astype(np.float32) + bias
    ref = np.where(ref > 0, ref, ref * np.float32(0.1)).astype(BF16).astype(np.float32).reshape(h, w, cout)
    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4, mode="bf16", a_rows=a_rows, a_c=c_store, a_w2=w2, a_kh=kh,
              c_rows=c_rows, c_shift=c_shift)
    cases.append((idx, h, w, cout, w2, kw, A, B, c_rows, N, ref))

arts = []
for idx, h, w, cout, w2, kw, *_ in cases:
    base = "/cache/rt/taps_" + "_".join(f"{v}" for v in kw.values())
    if not os.path.exists(base + ".bin"):
        t = time.perf_counter()
        cr.conv_rt.specialize(**kw).compile(xclbin_path=base + ".xclbin", inst_path=base + ".bin")
        print(f"conv{idx:02d}: compiled in {time.perf_counter()-t:.1f} s", flush=True)
    arts.append(base)

dev = npu_xrt.Device()
keep = []
for (idx, h, w, cout, w2, kw, A, B, c_rows, N, ref), base in zip(cases, arts):
    prog = npu_xrt.Program(dev, base + ".xclbin")
    seq = npu_xrt.Sequence(prog, base + ".bin")
    ab = npu_xrt.Buffer(dev, A.shape, BF16); ab.array[:] = A; ab.to_device()
    bb = npu_xrt.Buffer(dev, B.shape, BF16); bb.array[:] = B; bb.to_device()
    cb = npu_xrt.Buffer(dev, (c_rows, N), BF16); cb.array[:] = 0; cb.to_device()
    seq.run(ab, bb, cb)
    cb.from_device()
    got = cb.array[:(h + 2) * w2].reshape(h + 2, w2, N)[1:h + 1, 1:w + 1, :cout].astype(np.float32)
    t = time.perf_counter()
    for _ in range(10):
        seq.run(ab, bb, cb)
    dt = (time.perf_counter() - t) / 10
    d = np.abs(got - ref)
    tol = 0.02 * np.abs(ref).max()
    print(f"conv{idx:02d} taps k={kw['a_kh']}: max|diff| {d.max():.4g} (ref max {np.abs(ref).max():.3g}), "
          f"{(d > tol).mean()*100:.3f}% beyond tolerance, {dt*1e3:.2f} ms/call", flush=True)
    if (d > tol).any():
        bad = np.argwhere(d > tol)
        print("   first bad (y, x, c):", bad[:6].tolist())
    keep += [prog, seq, ab, bb, cb]

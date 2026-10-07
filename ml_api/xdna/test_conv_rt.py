"""Step 2b check: conv_rt's on-core epilogue against a host reference.

bf16: conv05 (normal, params on B). int8: conv13 (transposed, params on A).
Real activations and weights; artifacts compiled before any NPU context opens.
"""
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/w")
import aie.iron as iron
from aie.iron.device import from_name
import npu_xrt
from pack import BF16, a_src, b_src_rowmaj, epilogue_params_bf16, epilogue_params_i8, param_chunk
from yolo_host import YoloV2

spec = importlib.util.spec_from_file_location("conv_rt", "/w/conv_rt.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
iron.set_current_device(from_name("npu", n_cols=4))

net = YoloV2("/w/model.onnx")
sc = json.load(open("/w/scales.json"))
img = cv2.imread("/w/detached-1.jpg")
x = cv2.cvtColor(cv2.resize(img, (416, 416)), cv2.COLOR_BGR2RGB)
x = np.expand_dims(np.transpose(x, (2, 0, 1)).astype(np.float32), 0) / 255.0
inputs = {}
net.run(x, lambda layer, a: inputs.__setitem__(layer.index, a) or a @ layer.weight)


def ceil(v, a):
    return -(-v // a) * a


cases = []
# --- bf16, conv05: M=2704 K=1152 N=256, tiles 32x64x64
L = net.convs[5]; a = inputs[5]; M0, K0 = a.shape; N0 = L.weight.shape[1]
m, k, n = 32, 64, 64
M, K, N = ceil(M0, 8 * m), ceil(K0, k), ceil(N0, n * 4)
A = np.zeros((M, K), BF16); A[:M0, :K0] = a
W = np.zeros((k + K, N), BF16); W[k:k + K0, :N0] = L.weight
src = b_src_rowmaj(k, n, 8, 4)
for j in range(N // n):
    ch = slice(j * n, (j + 1) * n)
    bias = np.zeros(n, np.float32); bias[:max(0, min(n, N0 - j * n))] = L.bias[ch][:max(0, min(n, N0 - j * n))]
    W[:k, ch] = param_chunk(epilogue_params_bf16(bias), src, (k, n), BF16)
ref = A[:M0, :K0].astype(np.float32) @ W[k:k + K0, :N0].astype(np.float32) + L.bias
ref = np.where(ref > 0, ref, ref * np.float32(0.1)).astype(BF16).astype(np.float32)
cases.append(("conv05 bf16", "bf16", (M, K, N, m, k, n), A, W, (M, N), BF16, lambda c: c[:M0, :N0].astype(np.float32), ref))

# --- int8, conv13 transposed: channels x patches, tiles 64x64x48
L = net.convs[13]; a = inputs[13]; P0, K0 = a.shape; C0 = L.weight.shape[1]
m, k, n = 64, 64, 48
M, K, N = ceil(C0, 8 * m), ceil(K0, k), ceil(P0, n * 4)
s_in, zp_in = sc["s_in"]["13"], sc["zp_in"]["13"]
s_out, zp_out = sc["s_in"]["14"], sc["zp_in"]["14"]          # conv13 feeds conv14
s_w = np.maximum(np.abs(L.weight).max(axis=0), 1e-12) / 127.0
w_q = np.clip(np.round(L.weight / s_w), -127, 127).astype(np.int8)
x_q = np.clip(np.rint(a / s_in) + zp_in, -128, 127).astype(np.int8)
colsum = w_q.astype(np.int64).sum(axis=0)
scale_c = (s_in * s_w / s_out).astype(np.float32)
bias_c = ((L.bias - zp_in * colsum * s_in * s_w) / s_out).astype(np.float32)
A = np.zeros((M, k + K), np.int8); A[:C0, k:k + K0] = w_q.T
src = a_src(m, k, 4, 8)
ref_parts = []
for i in range(M // m):
    ch = slice(i * m, (i + 1) * m)
    sc_t = np.zeros(m); bi_t = np.zeros(m)
    valid = max(0, min(m, C0 - i * m))
    sc_t[:valid] = scale_c[ch][:valid]; bi_t[:valid] = bias_c[ch][:valid]
    prm, fx = epilogue_params_i8(sc_t, bi_t, zp_out)
    A[ch, :k] = param_chunk(prm, src, (m, k), np.int8)
    ref_parts.append(fx)
B = np.full((N, K), zp_in, np.int8); B[:P0, :K0] = x_q
acc = x_q.astype(np.int64) @ w_q.astype(np.int64)                 # [patches, channels]
mpos = np.concatenate([f[0] for f in ref_parts])[:C0].astype(np.int64)
mneg = np.concatenate([f[1] for f in ref_parts])[:C0].astype(np.int64)
bacc = np.concatenate([f[2] for f in ref_parts])[:C0].astype(np.int64)
shifts = np.concatenate([[f[3]] * m for f in ref_parts])[:C0]
yv = acc + bacc
prod = yv * np.where(yv > 0, mpos, mneg) + (zp_out * (2 ** shifts.astype(np.int64)))
q = np.where(prod >= 0, (prod + (2 ** (shifts - 1))) >> shifts, -((-prod + (2 ** (shifts - 1))) >> shifts))
ref = np.clip(q, -128, 127)
cases.append(("conv13 i8t", "i8t", (M, K, N, m, k, n), A, B, (M, N), np.int8, lambda c: c[:C0, :P0].T.astype(np.float32), ref))

arts = []
for name, mode, (M, K, N, m, k, n), *_ in cases:
    base = f"/cache/rt/conv_rt_{mode}_{m}x{k}x{n}_{M}_{K}_{N}"
    if not os.path.exists(base + ".bin"):
        t = time.perf_counter()
        cr.conv_rt.specialize(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4, mode=mode).compile(xclbin_path=base + ".xclbin", inst_path=base + ".bin")
        print(f"{name}: compiled in {time.perf_counter()-t:.1f} s", flush=True)
    arts.append(base)

dev = npu_xrt.Device()
keep = []
for (name, mode, dims, A, B, cshape, cdt, view, ref), base in zip(cases, arts):
    prog = npu_xrt.Program(dev, base + ".xclbin")
    seq = npu_xrt.Sequence(prog, base + ".bin")
    ab = npu_xrt.Buffer(dev, A.shape, A.dtype); ab.array[:] = A; ab.to_device()
    bb = npu_xrt.Buffer(dev, B.shape, B.dtype); bb.array[:] = B; bb.to_device()
    cb = npu_xrt.Buffer(dev, cshape, cdt)
    seq.run(ab, bb, cb)
    cb.from_device()
    got = view(cb.array)
    t = time.perf_counter()
    for _ in range(10):
        seq.run(ab, bb, cb)
    dt = (time.perf_counter() - t) / 10
    d = np.abs(got - ref)
    print(f"{name}: max|diff| {d.max():.4g} (ref max {np.abs(ref).max():.3g}), "
          f"{(d > (1 if mode == 'i8t' else 0.02 * np.abs(ref).max())).mean()*100:.2f}% beyond tolerance, {dt*1e3:.2f} ms/call", flush=True)
    bad = d > (1 if mode == "i8t" else 0.02 * np.abs(ref).max())
    if bad.any():
        Mv, Nv = bad.shape
        rows = np.where(bad.any(axis=1))[0]
        cols = np.where(bad.any(axis=0))[0]
        print("  bad rows: count", len(rows), "first", rows[:12], "mod 32 hist", np.bincount(rows % 32, minlength=32))
        print("  bad cols: count", len(cols), "first", cols[:20], "mod 64 hist", np.bincount(cols % 64, minlength=64))
        r0, c0 = rows[0], cols[0]
        print("  sample got", got[r0, :8], "\n  sample ref", ref[r0, :8])
    keep += [prog, seq, ab, bb, cb]

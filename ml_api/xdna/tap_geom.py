"""conv_rt tap mode on an arbitrary geometry with random data, checked against NumPy.

    python tap_geom.py H W C COUT KH m k n [timeout_s]
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/w")
import aie.iron as iron
from aie.iron.device import from_name
import npu_xrt
from pack import BF16, b_src_rowmaj, epilogue_params_bf16, param_chunk

H, W, C, COUT, KH, m, k, n = [int(a) for a in sys.argv[1:9]]
spec = importlib.util.spec_from_file_location("conv_rt", "/w/conv_rt.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
iron.set_current_device(from_name("npu", n_cols=4))


def ceil(v, a):
    return -(-v // a) * a


w2 = W + 2
M = ceil(H * w2, 8 * m)
K = KH * KH * C
N = ceil(COUT, n * 4)
a_rows = M + 2 * w2 + 2 + k
c_rows = max(w2 + 1 + M, (H + 2) * w2)
kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4, mode="bf16", a_rows=a_rows, a_c=C, a_w2=w2, a_kh=KH,
          c_rows=c_rows, c_shift=w2 + 1)
base = "/cache/rt/geom_" + "_".join(f"{v}" for v in kw.values())
if not os.path.exists(base + ".bin"):
    cr.conv_rt.specialize(**kw).compile(xclbin_path=base + ".xclbin", inst_path=base + ".bin")
if os.environ.get("COMPILE_ONLY"):
    sys.exit(0)

rng = np.random.default_rng(0)
act = (rng.random((H, W, C), dtype=np.float32) - 0.5).astype(BF16).astype(np.float32)
wgt = ((rng.random((COUT, C, KH, KH), dtype=np.float32) - 0.5) * 0.2).astype(BF16).astype(np.float32)
bias = (rng.random(COUT, dtype=np.float32) - 0.5)
A = np.zeros((a_rows, C), BF16)
A[:(H + 2) * w2].reshape(H + 2, w2, C)[1:-1, 1:-1] = act
B = np.zeros((k + K, N), BF16)
B[k:, :COUT] = wgt.transpose(2, 3, 1, 0).reshape(K, COUT)
src = b_src_rowmaj(k, n, 8, 4)
for j in range(N // n):
    bj = np.zeros(n, np.float32)
    v = max(0, min(n, COUT - j * n))
    bj[:v] = bias[j * n:j * n + v]
    B[:k, j * n:(j + 1) * n] = param_chunk(epilogue_params_bf16(bj), src, (k, n), BF16)
xp = np.pad(act, ((1, 1), (1, 1), (0, 0))) if KH == 3 else act
if KH == 3:
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(0, 1)).transpose(0, 1, 3, 4, 2).reshape(H * W, 9 * C)
else:
    win = act.reshape(H * W, C)
ref = win @ wgt.transpose(2, 3, 1, 0).reshape(K, COUT) + bias
ref = np.where(ref > 0, ref, 0.1 * ref).reshape(H, W, COUT)

dev = npu_xrt.Device()
prog = npu_xrt.Program(dev, base + ".xclbin")
seq = npu_xrt.Sequence(prog, base + ".bin")
ab = npu_xrt.Buffer(dev, A.shape, BF16); ab.array[:] = A; ab.to_device()
bb = npu_xrt.Buffer(dev, B.shape, BF16); bb.array[:] = B; bb.to_device()
cb = npu_xrt.Buffer(dev, (c_rows, N), BF16); cb.array[:] = 0; cb.to_device()
try:
    seq.run(ab, bb, cb)
except Exception as e:
    print(f"{H}x{W}x{C}->{COUT} k{KH} tiles {m}x{k}x{n} groups {N // n // 4}: FAIL {e}")
    sys.exit(1)
cb.from_device()
got = cb.array[:(H + 2) * w2].reshape(H + 2, w2, N)[1:-1, 1:-1, :COUT].astype(np.float32)
d = np.abs(got - ref)
print(f"{H}x{W}x{C}->{COUT} k{KH} tiles {m}x{k}x{n} groups {N // n // 4}: max|diff| {d.max():.3g} (ref max {np.abs(ref).max():.3g})")

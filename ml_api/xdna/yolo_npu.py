"""Obico's YOLOv2 on the XDNA1 NPU: conv_rt per layer, activations in NPU buffers.

conv00-12  bf16, implicit im2col (DMA taps) from padded flat buffers, epilogue
           (bias + leaky) on the cores, output straight into the next layer's
           padded buffer. The host re-zeroes the border cells the junk GEMM rows
           land on, and does the five 2x2 max-pools.
conv13-21  int8 (transposed GEMM, weights x patches): host-built int8 patches,
           requantising epilogue on the cores; the host transposes the output.
conv22     1x1 -> 30 channels, float on the CPU (5 M MACs).
reorg and the box-decoding tail run in ONNX Runtime, cut from the model.

    python yolo_npu.py --compile     # build every artifact (no NPU contexts open)
"""
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

import npu_xrt
from pack import a_src, b_src_rowmaj, epilogue_params_bf16, epilogue_params_i8, param_chunk

BF16 = ml_dtypes.bfloat16
HERE = os.path.dirname(os.path.abspath(__file__))
BF16_TILES = ((64, 32, 16), (64, 64, 32), (32, 64, 64))
I8_TILES = (64, 64, 48)
FIRST_INT8 = 13
COLS = 4
REORG_IN, REORG_OUT, CONCAT, TAIL_IN = "202", "213", "214", "218"


def ceil(v, a):
    return -(-v // a) * a


def conv_rt_module():
    spec = importlib.util.spec_from_file_location("conv_rt", os.path.join(HERE, "conv_rt.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Layer:
    index: int
    name_in: str
    name_out: str          # tensor after the activation
    weight: np.ndarray     # [Cout, Cin, kh, kw]
    bias: np.ndarray
    H: int = 0
    W: int = 0
    pool_after: bool = False
    mode: str = ""         # "bf16", "i8t" or "cpu"
    kw: dict = field(default_factory=dict)   # conv_rt compile parameters

    @property
    def cout(self):
        return self.weight.shape[0]

    @property
    def cin(self):
        return self.weight.shape[1]

    @property
    def kh(self):
        return self.weight.shape[2]


def parse(onnx_path):
    model = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes = list(model.graph.node)
    consumers = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    layers, hw = [], {"input": (416, 416)}
    for n in nodes:
        if n.op_type == "Conv":
            nxt = [c for c in consumers.get(n.output[0], []) if c.op_type == "LeakyRelu"]
            out = nxt[0].output[0] if nxt else n.output[0]
            h, w = hw[n.input[0]]
            layers.append(Layer(len(layers), n.input[0], out, inits[n.input[1]], inits[n.input[2]].astype(np.float32), h, w))
            hw[out] = (h, w)
        elif n.op_type == "MaxPool":
            h, w = hw[n.input[0]]
            hw[n.output[0]] = (h // 2, w // 2)
            [l for l in layers if l.name_out == n.input[0]][0].pool_after = True
        elif n.output[0] in (REORG_OUT, CONCAT):
            hw[n.output[0]] = (13, 13)
    ext = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(model))
    return layers, ext


def plan(layers):
    """conv_rt compile parameters per layer."""
    for l in layers:
        if l.index == len(layers) - 1:
            l.mode = "cpu"
            continue
        if l.index >= FIRST_INT8:
            l.mode = "i8t"
            m, k, n = I8_TILES
            K = l.cin * l.kh * l.kh
            l.kw = dict(M=ceil(l.cout, 8 * m), K=ceil(K, k), N=ceil(l.H * l.W, n * COLS), m=m, k=k, n=n,
                        n_aie_cols=COLS, mode="i8t")
            continue
        l.mode = "bf16"
        w2 = l.W + 2
        c_store = 4 if l.cin == 3 else l.cin
        best = None
        for m, k, n in BF16_TILES:
            if l.kh == 3 and c_store < k and l.cin != 3:
                continue
            if c_store >= k and c_store % k:
                continue
            M = ceil(l.H * w2, 8 * m)
            K = 3 * k if (l.kh == 3 and c_store < k) else l.kh * l.kh * c_store
            N = ceil(l.cout, n * COLS)
            cost = M * K * N * (1 + 16 / m + 16 / n)
            if best is None or cost < best[0]:
                best = (cost, (m, k, n, M, K, N))
        m, k, n, M, K, N = best[1]
        l.kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=COLS, mode="bf16",
                    a_rows=M + 2 * w2 + 2 + k, a_c=c_store, a_w2=w2, a_kh=l.kh,
                    c_rows=max(w2 + 1 + M, (l.H + 2) * w2), c_shift=w2 + 1)
    # A chained bf16 layer writes into its successor's input buffer, so the
    # buffer must cover both the successor's reads and this layer's writes.
    for l, nx in zip(layers, layers[1:]):
        if l.mode == "bf16" and nx.mode == "bf16" and not l.pool_after:
            rows = max(l.kw["c_rows"], nx.kw["a_rows"])
            l.kw["c_rows"] = nx.kw["a_rows"] = rows
            assert l.kw["N"] == nx.kw["a_c"], (l.index, l.kw["N"], nx.kw["a_c"])
    # Tap mode runs one column group (n * COLS output channels) per dispatch.
    for l in layers:
        if l.mode == "bf16":
            g = n_groups(l)
            l.group_kw = [dict(l.kw, N=l.kw["N"] // g, c_ld=l.kw["N"], c_col0=j * l.kw["N"] // g) for j in range(g)]
        elif l.mode == "i8t":
            l.group_kw = [l.kw]
    return layers


def n_groups(l):
    return l.kw["N"] // (l.kw["n"] * COLS)


def artifact_base(cache_dir, kw):
    return os.path.join(cache_dir, "conv_rt_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())))


def compile_all(layers, cache_dir):
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu", n_cols=COLS))
    cr = conv_rt_module()
    os.makedirs(cache_dir, exist_ok=True)
    for l in layers:
        if l.mode == "cpu":
            continue
        for kw in l.group_kw:
            base = artifact_base(cache_dir, kw)
            if not os.path.exists(base + ".bin"):
                print(f"compiling conv{l.index:02d} {kw}", flush=True)
                cr.conv_rt.specialize(**kw).compile(xclbin_path=base + ".xclbin", inst_path=base + ".bin")


def program_key(kw):
    return (kw["mode"], kw["m"], kw["k"], kw["n"])


def maxpool_bf16(src_hwc, out_hwc):
    """2x2/2 max-pool of a bf16 view into another bf16 view.

    bf16 is the top half of a float32, so widening the bits (a shift, no
    rounding) lets NumPy compare natively; the result is exact. Works on
    strided views directly, no contiguous copy first.
    """
    u = src_hwc.view(np.uint16)
    f = lambda a: (a.astype(np.uint32) << 16).view(np.float32)
    m = np.maximum(np.maximum(f(u[0::2, 0::2]), f(u[0::2, 1::2])), np.maximum(f(u[1::2, 0::2]), f(u[1::2, 1::2])))
    out_hwc.view(np.uint16)[:] = (m.view(np.uint32) >> 16).astype(np.uint16)


class YoloNpu:
    def __init__(self, onnx_path, scales, cache_dir):
        self.layers, ext = parse(onnx_path)
        plan(self.layers)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.reorg = ort.InferenceSession(ext.extract_model([REORG_IN], [REORG_OUT]).SerializeToString(), opts,
                                          providers=["CPUExecutionProvider"])
        self.tail = ort.InferenceSession(ext.extract_model([TAIL_IN], ["boxes", "confs"]).SerializeToString(), opts,
                                         providers=["CPUExecutionProvider"])
        self.s_in = {int(k): v for k, v in scales["s_in"].items()}
        self.zp_in = {int(k): int(v) for k, v in scales["zp_in"].items()}
        self.dev = npu_xrt.Device()
        self.programs = {}
        L = self.layers
        # which layer's scale an int8 output is quantised to (its consumer's input)
        self.out_q = {13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 21, 20: 21, 21: 22}
        self.bufs = {}
        for l in L:
            if l.mode == "cpu":
                continue
            seqs = []
            for kw in l.group_kw:
                base = artifact_base(cache_dir, kw)
                key = program_key(kw)
                if key not in self.programs:
                    self.programs[key] = npu_xrt.Program(self.dev, base + ".xclbin")
                seqs.append(npu_xrt.Sequence(self.programs[key], base + ".bin"))
            if l.mode == "bf16":
                self._setup_bf16(l, seqs)
            else:
                self._setup_i8(l, seqs[0])

    # ---------------------------------------------------------------- setup
    def _setup_bf16(self, l, seqs):
        kw = l.kw
        m, k, n, K, N = kw["m"], kw["k"], kw["n"], kw["K"], kw["N"]
        cin, cout, kh = l.cin, l.cout, l.kh
        c_store = kw["a_c"]
        if kh == 3 and c_store >= k:
            Wk = l.weight.transpose(2, 3, 1, 0).reshape(K, cout)
        elif kh == 3:
            Wk = np.zeros((K, cout), np.float32)
            for dy in range(3):
                for dx in range(3):
                    Wk[dy * k + dx * c_store: dy * k + dx * c_store + cin] = l.weight[:, :, dy, dx].T
        else:
            Wk = l.weight[:, :, 0, 0].T
        full = np.zeros((k + K, N), BF16)
        full[k:, :cout] = Wk
        src = b_src_rowmaj(k, n, 8, 4)
        for j in range(N // n):
            bj = np.zeros(n, np.float32)
            valid = max(0, min(n, cout - j * n))
            bj[:valid] = l.bias[j * n:j * n + valid]
            full[:k, j * n:(j + 1) * n] = param_chunk(epilogue_params_bf16(bj), src, (k, n), BF16)
        g = len(seqs)
        wbs = []
        for j in range(g):     # one weight panel (and dispatch) per column group
            wb = npu_xrt.Buffer(self.dev, (k + K, N // g), BF16)
            wb.array[:] = full[:, j * N // g:(j + 1) * N // g]
            wb.to_device()
            wbs.append(wb)
        prev = self.bufs.get(l.index - 1)
        if prev is not None and prev["chain"]:
            a = prev["c"]                       # the previous layer writes our padded input
        else:
            a = npu_xrt.Buffer(self.dev, (kw["a_rows"], c_store), BF16)
            a.array[:] = 0
            a.to_device()
        nx = self.layers[l.index + 1]
        chain = nx.mode == "bf16" and not l.pool_after
        c = npu_xrt.Buffer(self.dev, (kw["c_rows"], N), BF16)
        c.array[:] = 0
        c.to_device()
        self.bufs[l.index] = dict(seqs=seqs, a=a, ws=wbs, c=c, chain=chain)

    def _setup_i8(self, l, seq):
        kw = l.kw
        m, k, n, M, K, N = kw["m"], kw["k"], kw["n"], kw["M"], kw["K"], kw["N"]
        cout, i = l.cout, l.index
        Wk = l.weight.transpose(2, 3, 1, 0).reshape(-1, cout)     # (dy, dx, c) x Cout
        k_real = Wk.shape[0]
        s_w = np.maximum(np.abs(Wk).max(axis=0), 1e-12) / 127.0
        w_q = np.clip(np.round(Wk / s_w), -127, 127).astype(np.int8)
        s_in, zp_in = self.s_in[i], self.zp_in[i]
        o = self.out_q[i]
        s_out, zp_out = self.s_in[o], self.zp_in[o]
        colsum = w_q.astype(np.int64).sum(axis=0)
        a_c = (s_in * s_w / s_out)
        b_c = (l.bias - zp_in * colsum * s_in * s_w) / s_out
        wb = npu_xrt.Buffer(self.dev, (M, k + K), np.int8)
        wb.array[:] = 0
        wb.array[:cout, k:k + k_real] = w_q.T
        src = a_src(m, k, 4, 8)
        for t in range(M // m):
            ch = slice(t * m, (t + 1) * m)
            valid = max(0, min(m, cout - t * m))
            at = np.zeros(m); bt = np.zeros(m)
            at[:valid] = a_c[t * m:t * m + valid]
            bt[:valid] = b_c[t * m:t * m + valid]
            prm, _ = epilogue_params_i8(at, bt, zp_out)
            wb.array[ch, :k] = param_chunk(prm, src, (m, k), np.int8)
        wb.to_device()
        b = npu_xrt.Buffer(self.dev, (N, K), np.int8)
        b.array[:] = zp_in
        c = npu_xrt.Buffer(self.dev, (M, N), np.int8)
        self.bufs[i] = dict(seq=seq, w=wb, b=b, c=c, zp_in=zp_in, k_real=k_real)

    # ------------------------------------------------------------------ run
    def _zero_junk(self, buf, H, W):
        """Border cells of a padded buffer that the shifted GEMM output overwrote."""
        w2 = W + 2
        buf.from_device()
        a = buf.array
        grid = a[:(H + 2) * w2].reshape(H + 2, w2, -1)
        grid[1:, 0] = 0
        grid[1:, w2 - 1] = 0
        a[(H + 1) * w2:] = 0
        buf.to_device()

    def _i8_patches(self, act_q, l):
        """int8 activation [H, W, C] -> the layer's B buffer rows (patches x (dy, dx, c))."""
        e = self.bufs[l.index]
        H, W, C = act_q.shape
        if l.kh == 1:
            e["b"].array[:H * W, :C] = act_q.reshape(H * W, C)
        else:
            xp = np.full((H + 2, W + 2, C), e["zp_in"], np.int8)
            xp[1:-1, 1:-1] = act_q
            win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(0, 1))   # [H, W, C, 3, 3]
            e["b"].array[:H * W, :9 * C] = win.transpose(0, 1, 3, 4, 2).reshape(H * W, 9 * C)
        e["b"].to_device()

    def _run_i8(self, l):
        e = self.bufs[l.index]
        e["seq"].run(e["w"], e["b"], e["c"])
        e["c"].from_device()
        return np.ascontiguousarray(e["c"].array[:l.cout, :l.H * l.W].T).reshape(l.H, l.W, l.cout)

    def _quant(self, act_bf16, i):
        x = act_bf16.astype(np.float32)
        return np.clip(np.rint(x / self.s_in[i]) + self.zp_in[i], -128, 127).astype(np.int8)

    def _dequant(self, q, i):
        return (q.astype(np.float32) - self.zp_in[i]) * self.s_in[i]

    def run(self, image_nchw):
        L, B = self.layers, self.bufs
        # conv00 input: the image, 3 channels stored as 4, in the padded layout
        l0 = L[0]
        a0 = B[0]["a"]
        w2 = l0.W + 2
        a0.array[:(l0.H + 2) * w2].reshape(l0.H + 2, w2, 4)[1:-1, 1:-1, :3] = image_nchw[0].transpose(1, 2, 0)
        a0.to_device()
        act = {}
        trace = os.environ.get("YOLO_TRACE")
        prof = self.prof = {} if os.environ.get("YOLO_PROFILE") else None
        import time
        t_last = time.perf_counter()

        def mark(name):
            nonlocal t_last
            if prof is not None:
                t = time.perf_counter()
                prof[name] = prof.get(name, 0.0) + t - t_last
                t_last = t
        mark("input")
        for l in L:
            i = l.index
            if trace:
                print(f"run conv{i:02d} {l.mode}", flush=True)
            if l.mode == "bf16":
                e = B[i]
                for seq, wb in zip(e["seqs"], e["ws"]):
                    seq.run(e["a"], wb, e["c"])
                mark(f"npu{i:02d}")
                if e["chain"]:
                    self._zero_junk(e["c"], l.H, l.W)
                    mark(f"host{i:02d}")
                    continue
                e["c"].from_device()
                w2 = l.W + 2
                out = e["c"].array[:(l.H + 2) * w2].reshape(l.H + 2, w2, -1)[1:-1, 1:-1, :l.cout]
                if l.pool_after:
                    if i == 12:
                        act["pool12"] = np.empty((l.H // 2, l.W // 2, l.cout), BF16)
                        maxpool_bf16(out, act["pool12"])
                    else:
                        nx = B[i + 1]["a"]
                        h2, wp2 = l.H // 2, l.W // 2 + 2
                        maxpool_bf16(out, nx.array[:(h2 + 2) * wp2].reshape(h2 + 2, wp2, -1)[1:-1, 1:-1, :l.cout])
                        nx.to_device()
                if i == 12:
                    act["c12"] = np.ascontiguousarray(out)
                    self._i8_patches(self._quant(act["pool12"], 13), L[13])
                    self._i8_patches(self._quant(act["c12"], 20), L[20])
                mark(f"host{i:02d}")
            elif l.mode == "i8t":
                e = B[i]
                e["seq"].run(e["w"], e["b"], e["c"])
                mark(f"npu{i:02d}")
                e["c"].from_device()
                y = np.ascontiguousarray(e["c"].array[:l.cout, :l.H * l.W].T).reshape(l.H, l.W, l.cout)
                if i in (13, 14, 15, 16, 17, 18):
                    self._i8_patches(y, L[i + 1])
                elif i == 19:
                    act["c19"] = y
                elif i == 20:
                    r = self.reorg.run(None, {REORG_IN: self._dequant(y, 21).transpose(2, 0, 1)[None]})[0][0]
                    reorg_q = np.clip(np.rint(r.transpose(1, 2, 0) / self.s_in[21]) + self.zp_in[21], -128, 127).astype(np.int8)
                    self._i8_patches(np.concatenate([reorg_q, act["c19"]], axis=2), L[21])
                elif i == 21:
                    act["c21"] = self._dequant(y, 22)
                mark(f"host{i:02d}")
            else:   # conv22 on the CPU
                x = act["c21"].reshape(-1, l.cin)
                head = (x @ l.weight[:, :, 0, 0].T + l.bias).reshape(l.H, l.W, l.cout)
                res = self.tail.run(None, {TAIL_IN: np.ascontiguousarray(head.transpose(2, 0, 1)[None])})
                mark("conv22+tail")
                return res


if __name__ == "__main__":
    if "--compile" in sys.argv:
        layers, _ = parse(os.environ.get("MODEL", "/w/model.onnx"))
        plan(layers)
        compile_all(layers, os.environ.get("RT_CACHE", "/cache/rt"))
        for l in layers:
            print(f"conv{l.index:02d} {l.mode:4} {l.H}x{l.W} {l.cin}->{l.cout} k{l.kh} pool={l.pool_after} "
                  f"{ {k: v for k, v in l.kw.items() if k in ('M', 'K', 'N', 'm', 'k', 'n')} }")

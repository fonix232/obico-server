"""Obico's YOLOv2 with every convolution as a matrix multiply on a pluggable backend.

The ONNX graph is walked in order. Conv (+ the LeakyRelu that follows it),
MaxPool and channel Concat run here; the convolutions go to `matmul` as
im2col GEMMs. The two pieces that are neither — the reorg (Reshape/Transpose
chain) and the box-decoding tail — are cut out of the same ONNX model and run
by ONNX Runtime, so their numerics are the model's own.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

REORG_IN, REORG_OUT = "202", "213"
TAIL_IN = "218"


@dataclass
class ConvLayer:
    index: int          # position among the convolutions, 0-based
    name: str           # ONNX output name of the Conv
    weight: np.ndarray  # [K, N]: K ordered (Cin, kh, kw) to match im2col below
    bias: np.ndarray    # [N]
    k: int              # kernel size (1 or 3)
    leaky: float        # LeakyRelu alpha, or None for a linear output
    out_name: str       # tensor name after the activation


def im2col(x: np.ndarray, k: int) -> np.ndarray:
    """[H, W, C] -> [H*W, C*k*k], stride 1, same padding."""
    h, w, c = x.shape
    if k == 1:
        return x.reshape(h * w, c)
    p = k // 2
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(0, 1))  # [H, W, C, k, k]
    return win.reshape(h * w, c * k * k)


class YoloV2:
    def __init__(self, onnx_path: str):
        model = onnx.load(onnx_path)
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        # The extractor needs value_info for the cut points, which the export lacks.
        ext = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(model))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.reorg = ort.InferenceSession(ext.extract_model([REORG_IN], [REORG_OUT]).SerializeToString(), opts, providers=["CPUExecutionProvider"])
        self.tail = ort.InferenceSession(ext.extract_model([TAIL_IN], ["boxes", "confs"]).SerializeToString(), opts, providers=["CPUExecutionProvider"])

        nodes = list(model.graph.node)
        consumers: Dict[str, List[onnx.NodeProto]] = {}
        for n in nodes:
            for i in n.input:
                consumers.setdefault(i, []).append(n)

        self.steps = []
        self.convs: List[ConvLayer] = []
        self.input_hw = None
        for n in nodes:
            if n.op_type == "Conv":
                a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
                w = inits[n.input[1]]  # [Cout, Cin, kh, kw]
                cout, cin, kh, kw = w.shape
                assert kh == kw and a.get("strides", [1, 1]) == [1, 1] and a.get("group", 1) == 1
                assert a.get("pads", [0] * 4) == [kh // 2] * 4
                nxt = [c for c in consumers.get(n.output[0], []) if c.op_type == "LeakyRelu"]
                alpha = onnx.helper.get_attribute_value(nxt[0].attribute[0]) if nxt else None
                layer = ConvLayer(
                    index=len(self.convs),
                    name=n.output[0],
                    weight=np.ascontiguousarray(w.transpose(1, 2, 3, 0).reshape(cin * kh * kw, cout)).astype(np.float32),
                    bias=inits[n.input[2]].astype(np.float32),
                    k=kh,
                    leaky=alpha,
                    out_name=nxt[0].output[0] if nxt else n.output[0],
                )
                self.convs.append(layer)
                self.steps.append(("conv", n.input[0], layer))
            elif n.op_type == "MaxPool":
                self.steps.append(("maxpool", n.input[0], n.output[0]))
            elif n.op_type == "Concat" and n.output[0] == "214":
                self.steps.append(("concat", list(n.input), n.output[0]))
            elif n.op_type == "LeakyRelu" and n.output[0] == REORG_IN:
                self.steps.append(("reorg", REORG_IN, REORG_OUT))

    def run(self, image_nchw: np.ndarray, matmul: Callable[[ConvLayer, np.ndarray], np.ndarray]):
        """image_nchw: [1, 3, H, W] float32 in [0, 1]. Returns (boxes, confs) like the ONNX model."""
        t: Dict[str, np.ndarray] = {"input": np.ascontiguousarray(image_nchw[0].transpose(1, 2, 0))}
        for step in self.steps:
            kind = step[0]
            if kind == "conv":
                _, src, layer = step
                x = t[src]
                h, w, _ = x.shape
                y = matmul(layer, im2col(x, layer.k)) + layer.bias
                if layer.leaky is not None:
                    y = np.where(y > 0, y, y * np.float32(layer.leaky))
                t[layer.out_name] = y.reshape(h, w, -1)
            elif kind == "maxpool":
                _, src, dst = step
                x = t[src]
                h, w, c = x.shape
                t[dst] = x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))
            elif kind == "reorg":
                x = t[REORG_IN].transpose(2, 0, 1)[None]
                t[REORG_OUT] = self.reorg.run(None, {REORG_IN: np.ascontiguousarray(x)})[0][0].transpose(1, 2, 0)
            elif kind == "concat":
                _, srcs, dst = step
                t[dst] = np.concatenate([t[s] for s in srcs], axis=2)
        head = np.ascontiguousarray(t[TAIL_IN].transpose(2, 0, 1)[None])
        return self.tail.run(None, {TAIL_IN: head})


def numpy_matmul(layer: ConvLayer, a: np.ndarray) -> np.ndarray:
    return a @ layer.weight

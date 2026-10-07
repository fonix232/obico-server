"""The ONNX model on an AMD XDNA1 NPU (Phoenix / Hawk Point), with a CPU fallback.

The network itself runs in xdna/yolo_npu.py: every convolution but the last
on the NPU through precompiled mlir-aie programs (xdna/conv_rt.py), the early
layers in bf16 and the deep ones in calibrated int8. Pre- and post-processing
are OnnxNet's own. If the NPU ever fails a frame, that frame and every later
one go to ONNX Runtime on the CPU instead: a wedged NPU must not stop failure
detection.
"""
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from lib.meta import Meta
from lib.onnx import OnnxNet, post_processing

XDNA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xdna")
sys.path.insert(0, XDNA_DIR)
from yolo_npu import YoloNpu  # noqa: E402  (needs XDNA_DIR on the path)

RT_CACHE = os.environ.get("XDNA_RT_CACHE", "/opt/xdna/rt")
SCALES = os.environ.get("XDNA_SCALES", os.path.join(XDNA_DIR, "scales.json"))


class XdnaNet:
    meta: Meta

    def __init__(self, onnx_path: str, meta_path: str, use_gpu: bool):
        if not use_gpu:
            raise Exception('XdnaNet only runs on the NPU')
        if not os.path.exists('/dev/accel/accel0'):
            raise Exception('no NPU passed in (/dev/accel/accel0)')
        self.onnx_path = onnx_path
        self.meta_path = meta_path
        self.meta = Meta(meta_path)
        with open(SCALES) as f:
            self.npu = YoloNpu(onnx_path, json.load(f), RT_CACHE)
        self.input_h = self.input_w = 416
        self.cpu = None
        # One frame through the NPU now: a broken setup fails here, at boot,
        # where load_net() falls through to the next net.
        self.npu.run(np.zeros((1, 3, self.input_h, self.input_w), np.float32))
        print(f'XdnaNet on the NPU: {len(self.npu.programs)} programs')

    def detect(self, meta, image, alt_names, thresh=.5, hier_thresh=.5, nms=.45, debug=False) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        if self.cpu is not None:
            return self.cpu.detect(meta, image, alt_names, thresh, hier_thresh, nms, debug)
        width = image.shape[1]
        height = image.shape[0]

        # Input, prepared exactly as OnnxNet.detect() prepares it
        resized = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        try:
            outputs = self.npu.run(img_in)
        except Exception as e:
            print(f'XdnaNet: NPU failed ({e}); ONNX Runtime on the CPU from now on')
            self.cpu = OnnxNet(self.onnx_path, self.meta_path, False)
            return self.cpu.detect(meta, image, alt_names, thresh, hier_thresh, nms, debug)

        detections = post_processing(outputs, width, height, thresh, nms, meta.names)
        return detections[0]

"""The ONNX model, run by PyTorch on a ROCm or CUDA GPU.

ONNX Runtime has no supported execution provider for consumer AMD GPUs: AMD
supports its MIGraphX builds on Instinct parts. AMD does publish ROCm builds of
PyTorch per GPU target, Radeon iGPUs included, so this converts the same ONNX
graph into a torch module with onnx2torch and runs it on whatever torch.cuda
reports (a ROCm build exposes HIP devices through the CUDA API). Pre- and
post-processing are OnnxNet's own, so both nets return the same detections for
the same frame.
"""
from typing import List, Tuple

import cv2
import numpy as np
import onnx
import torch
from onnx2torch import convert

from lib.meta import Meta
from lib.onnx import post_processing


class TorchOnnxNet:
    meta: Meta

    def __init__(self, onnx_path: str, meta_path: str, use_gpu: bool):
        # On a CPU, ONNX Runtime is the faster of the two, and it is the next
        # entry in load_net()'s priority list.
        if not use_gpu:
            raise Exception('TorchOnnxNet only runs on a GPU')
        if not torch.cuda.is_available():
            raise Exception('PyTorch sees no ROCm/CUDA device')

        model = onnx.load(onnx_path)
        dims = model.graph.input[0].type.tensor_type.shape.dim
        self.input_h = dims[2].dim_value
        self.input_w = dims[3].dim_value

        self.device = torch.device('cuda')
        self.net = convert(model).eval().to(self.device)
        self.meta = Meta(meta_path)

        # The first pass compiles and loads the kernels, which takes seconds on
        # ROCm. Pay that here, inside worker boot, rather than on the first
        # request, whose caller is waiting on it.
        with torch.inference_mode():
            self.net(torch.zeros((1, 3, self.input_h, self.input_w), device=self.device))
        torch.cuda.synchronize()

        backend = f'HIP {torch.version.hip}' if torch.version.hip else f'CUDA {torch.version.cuda}'
        print(f'TorchOnnxNet on {torch.cuda.get_device_name(0)} (torch {torch.__version__}, {backend})')

    def detect(self, meta, image, alt_names, thresh=.5, hier_thresh=.5, nms=.45, debug=False) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        width = image.shape[1]
        height = image.shape[0]

        # Input, prepared exactly as OnnxNet.detect() prepares it
        resized = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        with torch.inference_mode():
            outputs = self.net(torch.from_numpy(img_in).to(self.device))

        # post_processing() moves tensors to the CPU itself.
        detections = post_processing(outputs, width, height, thresh, nms, meta.names)
        return detections[0]

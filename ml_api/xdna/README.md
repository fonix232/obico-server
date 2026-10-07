# Obico ML API on an AMD XDNA1 NPU

Runs Obico's YOLOv2 failure-detection model on the NPU of Ryzen 7040/8040 ("Phoenix" / "Hawk Point", mlir-aie target `npu1`) through programs built with IRON (mlir-aie) and the Peano compiler. `../Containerfile.xdna` builds them at image build time; the runtime image needs only XRT and `/dev/accel/accel0`.

Measured on a Ryzen 9 8945HS (Linux 6.18 in-tree `amdxdna`, firmware 1.5.2): 57 ms and 14 ms of CPU per frame, against 30 ms and 242 ms of CPU for ONNX Runtime on the same machine. Detections match the fp32 model to within a mean top-score drift of 0.006 on 30 frames.

## How it runs

- `conv_rt.py` — one GEMM design for every layer, derived from mlir-aie's `whole_array`. Loop counts arrive as runtime parameters, so there is one resident program per tile shape (four in all; NPU1 allows few hardware contexts). The epilogue parameters ride as the first chunk of the weight operand, and each core runs `kernels/epilogue.cc` on its accumulator: bias + leaky ReLU with bf16 output, or fixed-point requantisation to int8.
- Layers 0-12 run in bf16 with implicit im2col: activations stay in padded, flattened NPU buffers, every 3x3 tap is a shifted DMA read, and each layer writes straight into the next one's input buffer.
- Layers 13-21 run in int8: they are the least sensitive layers and hold most of the arithmetic. `scales.json` comes from `quant.py`, calibrated on real printer frames. Layer 22, the reorg and the box decoding stay on the CPU.
- `yolo_npu.py --compile` builds every program without a device; `lib/xdna_net.py` falls back to ONNX Runtime on the CPU if the NPU fails.

## Limits found on NPU1 (mlir-aie 1.4.3, XRT 2.26)

- More than four hardware contexts made the driver reject commands (`EXEC_CMD` EINVAL); so did compiling while contexts were open.
- `whole_array` with int8 inputs and a column-major C returns wrong results.
- Repeating an A panel fill for a second column group hung the array; wide layers run as one dispatch per column group instead.
- 16-lane bf16 vector loads need 32-byte alignment.

The other scripts here (`test_*.py`, `tap_geom.py`, `run_full.py`, `profile_full.py`, `quant.py`) are the development harness: they run in the `Containerfile.toolchain` image with this directory mounted at `/w`.

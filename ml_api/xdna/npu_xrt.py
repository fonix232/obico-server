"""Minimal direct XRT runtime for mlir-aie NPU designs.

IRON's runtime caches hw_contexts and instruction BOs in LRUs sized to the
NPU1 context limit (6) and drains them on Phoenix; with one program shared by
many instruction streams that thrashes and leads to EXEC_CMD EINVAL. Here the
caller owns everything: one hw_context + kernel per program, one instruction
BO per sequence, persistent data BOs, all created once.
"""
import numpy as np
import pyxrt

_TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
_FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


class Device:
    def __init__(self, index=0):
        self.dev = pyxrt.device(index)


class Program:
    """One xclbin loaded into one hardware context."""

    def __init__(self, device, xclbin_path, kernel_name="MLIR_AIE"):
        self.device = device
        self.xclbin = pyxrt.xclbin(str(xclbin_path))
        device.dev.register_xclbin(self.xclbin)
        self.context = pyxrt.hw_context(device.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.context, kernel_name)


class Sequence:
    """An instruction stream (runtime sequence) for a Program."""

    def __init__(self, program, insts_path):
        insts = np.fromfile(str(insts_path), dtype=np.uint32)
        self.program = program
        self.nbytes = insts.nbytes
        self.bo = pyxrt.bo(program.device.dev, insts.nbytes, pyxrt.bo.cacheable, program.kernel.group_id(1))
        np.frombuffer(self.bo.map(), dtype=np.uint32)[:] = insts
        self.bo.sync(_TO)

    def run(self, *buffers):
        h = self.program.kernel(3, self.bo, self.nbytes, *[b.bo for b in buffers])
        state = h.wait()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"NPU command ended in {state}")


class Buffer:
    """A host-mapped NPU buffer; .array is a writable NumPy view of it."""

    def __init__(self, device, shape, dtype):
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        self.bo = pyxrt.bo(device.dev, nbytes, pyxrt.bo.host_only, 0)
        self.array = np.frombuffer(self.bo.map(), dtype=dtype, count=int(np.prod(shape))).reshape(shape)

    def to_device(self):
        self.bo.sync(_TO)

    def from_device(self):
        self.bo.sync(_FROM)

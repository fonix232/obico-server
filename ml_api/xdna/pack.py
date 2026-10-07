"""Host-side packing for conv_rt: the epilogue parameter chunk.

The chunk travels as one more element of the A or B stream, so the memtile
applies that stream's L2->L1 blocking to it. Packing scatters the wanted
bytes through the inverse of that blocking; the core then sees them in order.
"""
import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16


def _src_index(sizes_strides):
    """L2 source element for each L1 position of a 4-D (size, stride) pattern."""
    (n0, s0), (n1, s1), (n2, s2), (n3, s3) = sizes_strides
    i0, i1, i2, i3 = np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    return (i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3).reshape(-1)


def a_src(m, k, r, s):
    return _src_index([(m // r, r * k), (k // s, s), (r, k), (s, 1)])


def b_src_rowmaj(k, n, s, t):
    return _src_index([(k // s, s * n), (n // t, t), (s, n), (t, 1)])


def epilogue_params_bf16(bias, alpha=0.1):
    """bf16 layout read by epi_bf16: header floats, then 16-lane bias vectors per 4-channel block."""
    n = len(bias)
    head = np.array([0.0, alpha, 0, 0, 0, 0, 0, 0], np.float32)   # 32 bytes: keeps bias_rep 32-byte aligned
    rep = np.repeat(np.asarray(bias, np.float32).reshape(n // 4, 1, 4), 4, axis=1).reshape(-1)   # [n/4][4 rows][4 ch]
    return np.frombuffer(head.tobytes() + rep.astype(BF16).tobytes(), np.uint8)


def epilogue_params_i8(a, b, zp_out, alpha=0.1):
    """int8 layout read by epi_i8: q = round((acc + bias) * mult >> shift) + zp.

    a, b: per channel, the float requantisation y = acc * a + b. The bias moves
    into accumulator units; one shift per tile keeps the largest multiplier
    within int16.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    amax = max(float(np.abs(a).max()), 1e-30)
    shift = int(np.clip(np.floor(np.log2(32767.0 / amax)), 0, 62))
    mpos = np.clip(np.rint(a * 2.0 ** shift), -32768, 32767).astype(np.int16)
    mneg = np.clip(np.rint(a * alpha * 2.0 ** shift), -32768, 32767).astype(np.int16)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_acc = np.where(np.abs(a) > 0, b / a, 0.0)
    bias_acc = np.clip(np.rint(bias_acc), -2**31, 2**31 - 1).astype(np.int32)
    head = np.array([int(zp_out), shift, 0, 0], np.int32)
    return np.frombuffer(head.tobytes() + bias_acc.tobytes() + mpos.tobytes() + mneg.tobytes(), np.uint8), (mpos, mneg, bias_acc, shift)


def param_chunk(params, src, shape, dtype):
    """One (rows, cols) chunk in DDR order whose L1 image starts with the `params` bytes."""
    want = np.frombuffer(np.asarray(params, np.uint8).tobytes(), dtype=dtype)
    l1 = np.zeros(int(np.prod(shape)), dtype)
    assert len(want) <= l1.size, "parameters do not fit in one chunk"
    l1[:len(want)] = want
    ddr = np.zeros_like(l1)
    ddr[src] = l1
    return ddr.reshape(shape)

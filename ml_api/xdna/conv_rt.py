"""Conv GEMM with an on-core epilogue, one resident program per tile config.

Derived from mlir-aie's whole_array (v1.4.3): same 4 x n_aie_cols core grid,
memtile splits/joins and runtime-sequence structure. Changes:

* Loop counts come from per-core RTPs behind a WorkerRuntimeBarrier
  (tiles per core, K // k), so the program only depends on the tiles.
* Each output tile starts with one extra chunk on the operand that carries
  the output channels -- B (weights) for "bf16", A (weights) for "i8t" --
  holding the epilogue parameters (see kernels/epilogue.cc). The host
  pre-permutes it against the L2->L1 blocking so the core reads it as floats.
* The core accumulates into a local buffer and the epilogue writes the output
  tile: bias + leaky ReLU, as bf16 ("bf16") or requantised int8 ("i8t").

Implicit im2col ("bf16" only, a_kh > 0): A is a padded, flattened activation
buffer [a_rows, a_c] -- row (y+1)*a_w2 + (x+1) holds pixel (y, x), borders
zero -- and GEMM row o = y*a_w2 + x (x in [0, a_w2), the last two columns
junk) reads its 3x3 taps at rows o + dy*a_w2 + dx. K is ordered (dy, dx, c);
for one dy the (dx, c) chunks are contiguous, so a row block's whole A panel
is one 4-D pattern [dy][dx*c/k][m rows][k]. With a_c < k (the 3-channel input,
stored as 4) one chunk per dy covers all (dx, c) pairs. 1x1 layers read row
o + a_w2 + 1. With c_rows > 0, C is written at row offset c_shift of a
[c_rows, N] buffer -- the next layer's padded input when c_shift = a_w2 + 1;
the junk columns land on its border cells, which the host re-zeroes.

Modes:
  bf16  A = patches [M,K] bf16, B = [k params + K, N] bf16, C = [M,N] bf16
  i8t   A = [M, k params + K] int8 (weights.T), B = patches as [N,K] int8
        (b_col_maj), C = [M,N] int8 row-major (channels x patches)
"""
import os

import ml_dtypes
import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    kernels,
)
from aie.iron.controlflow import range_

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
BF16 = ml_dtypes.bfloat16
MODES = {
    # mode: (in, acc, out, params on A?, b_col_maj)
    "bf16": (BF16, np.float32, BF16, False, 0),
    "i8t": (np.int8, np.int32, np.int8, True, 1),
}


def param_floats(mode, m, n):
    # bf16: 8-float header + 2n floats of bias vectors; int8: 4-word header + 2m words
    return 4 + 2 * m if MODES[mode][3] else 8 + 2 * n


def _tap_patterns(M, K, m, k, a_rows, a_c, a_w2, a_kh):
    """A pattern per row block for the implicit im2col (elements of the flat buffer)."""
    dims = [a_rows * a_c]
    taps = []
    for i in range(M // m):
        if a_kh == 3 and a_c >= k:
            assert a_c % k == 0 and K == 9 * a_c
            sizes, strides, off = [3, 3 * a_c // k, m, k], [a_w2 * a_c, k, a_c, 1], i * m * a_c
        elif a_kh == 3:
            assert 3 * a_c <= k and K == 3 * k
            sizes, strides, off = [3, m, k], [a_w2 * a_c, a_c, 1], i * m * a_c
        else:
            assert a_kh == 1 and a_c % k == 0 and K == a_c
            sizes, strides, off = [a_c // k, m, k], [k, a_c, 1], (i * m + a_w2 + 1) * a_c
        extent = off + sum((sz - 1) * st for sz, st in zip(sizes, strides)) + 1
        assert extent <= dims[0], f"row block {i} reads past the input buffer ({extent} > {dims[0]})"
        taps.append(TensorAccessPattern(dims, off, sizes, strides))
    return taps


def _build(dev, M, K, N, m, k, n, n_aie_cols, mode, a_rows=0, a_c=0, a_w2=0, a_kh=0, c_rows=0, c_shift=0, c_ld=0, c_col0=0):
    dtype_in, dtype_acc, dtype_out, params_on_a, b_col_maj = MODES[mode]
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    KA = K + (k if params_on_a else 0)
    KB = K + (0 if params_on_a else k)

    matmul_kernel = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_acc,
                               b_col_maj=bool(b_col_maj), c_col_maj=False)
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert M % (m * n_aie_rows) == 0 and K % k == 0 and N % (n * n_aie_cols) == 0
    assert m % r == 0 and k % s == 0 and n % t == 0

    pf = param_floats(mode, m, n)
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    acc_ty = np.ndarray[(m * n,), np.dtype[dtype_acc]]   # flat, as the mm/zero kernels declare it
    out_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    prm_ty = np.ndarray[(pf,), np.dtype[np.float32]]
    flags = [f"-DDIM_M={m}", f"-DDIM_N={n}"]
    epi = ExternalFunction(
        "epi_i8" if params_on_a else "epi_bf16",
        source_file=os.path.join(KDIR, "epilogue.cc"),
        arg_types=[acc_ty, out_l1_ty, prm_ty],
        include_dirs=[KDIR],
        compile_flags=flags + ["-DEPI_I8" if params_on_a else "-DEPI_BF16"],
    )
    loadp = ExternalFunction(
        "load_params",
        source_file=os.path.join(KDIR, "params.cc"),
        arg_types=[A_l1_ty if params_on_a else B_l1_ty, prm_ty],
        include_dirs=[KDIR],
        compile_flags=[f"-DPARAM_FLOATS={pf}"],
    )

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    tap_mode = a_kh > 0
    assert not tap_mode or mode == "bf16"
    A_ty = np.ndarray[((a_rows * a_c) if tap_mode else (M * KA),), np.dtype[dtype_in]]
    B_ty = np.ndarray[(KB * N,), np.dtype[dtype_in]]
    c_ld = c_ld or N
    C_ty = np.ndarray[((c_rows * c_ld) if c_rows else (M * N),), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]

    A_l3l2_fifos, A_l2l1_fifos, B_l3l2_fifos, B_l2l1_fifos = [], [], [], []
    C_l1l2_fifos = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos = []
    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        a_dims: list[StreamDims] = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        A_l2l1_fifos.extend(a_l3l2.cons().split(
            [m * k * j for j in range(stop_row - start_row)],
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        ))
    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims: StreamDims = ([(n // t, t * k), (k // s, s), (t, k), (s, 1)] if b_col_maj
                              else [(k // s, s * n), (n // t, t), (s, n), (t, 1)])
        B_l2l1_fifos.append(b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims))
        c_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3_fifos.append(c_l2l3)
        c_tmp = c_l2l3.prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[out_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp[j])

    rtps = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{row}_{col}",
                    initial_value=np.array([0, 0], dtype=np.int32), use_write_rtp=True)
             for col in range(n_aie_cols)] for row in range(n_aie_rows)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]
    accs = [[Buffer(acc_ty, name=f"acc_{row}_{col}") for col in range(n_aie_cols)] for row in range(n_aie_rows)]
    prms = [[Buffer(prm_ty, name=f"prm_{row}_{col}") for col in range(n_aie_cols)] for row in range(n_aie_rows)]

    def core_fn(in_a, in_b, out_c, zero, matmul, epilogue, load_params, acc, prm, rtp, barrier):
        barrier.wait_for_value(1)
        n_tiles = rtp[0]
        k_iters = rtp[1]
        for _ in range_(n_tiles):
            if params_on_a:
                elem = in_a.acquire(1)
                load_params(elem, prm)
                in_a.release(1)
            else:
                elem = in_b.acquire(1)
                load_params(elem, prm)
                in_b.release(1)
            zero(acc)
            for _ in range_(k_iters):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, acc)
                in_a.release(1)
                in_b.release(1)
            elem_out = out_c.acquire(1)
            epilogue(acc, elem_out, prm)
            out_c.release(1)
        barrier.release_with_value(1)

    workers = Worker.grid(
        n_aie_rows, n_aie_cols,
        lambda row, col: Worker(core_fn, [
            A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(), C_l1l2_fifos[row][col].prod(),
            zero_kernel, matmul_kernel, epi, loadp, accs[row][col], prms[row][col], rtps[row][col], barriers[row][col],
        ], stack_size=0xD00),
    )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2
    n_col_groups = N // n // n_aie_cols
    if tap_mode:
        # One column group per dispatch: repeating the panel fill for a second
        # group hung the array (verified); wider layers are split by the host
        # into dispatches writing their channel range (c_ld, c_col0).
        assert n_A_tiles_per_shim == 1 and n_col_groups == 1, "tap mode takes one column group per dispatch"
        A_tiles = _tap_patterns(M, K, m, k, a_rows, a_c, a_w2, a_kh)
        a_fills = 1
    else:
        A_tiles = TensorTiler2D.group_tiler((M, KA), (m * n_A_tiles_per_shim, k), (1, KA // k),
                                            pattern_repeat=n_col_groups, prune_step=False)
        a_fills = 1
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler((N, KB), (n, k), tile_group_repeats=(N // n // n_aie_cols, KB // k),
                                           tile_group_steps=(n_aie_cols, 1), prune_step=False)
    else:
        B_tiles = TensorTiler2D.step_tiler((KB, N), (k, n), tile_group_repeats=(KB // k, N // n // n_aie_cols),
                                           tile_group_steps=(1, n_aie_cols), tile_group_col_major=True, prune_step=False)
    C_tiles = TensorTiler2D.step_tiler((M, N), (m * n_aie_rows, n), tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
                                       tile_group_steps=(1, n_aie_cols), prune_step=False)
    if c_rows:
        # Tiled over this dispatch's own [M, N]; moved into the [c_rows, c_ld]
        # buffer at (c_shift, c_col0). Tiling the wide matrix instead yields
        # patterns for every column of it, out of step with the sequence below.
        assert c_shift + M <= c_rows and c_col0 + N <= c_ld

        def _move(tap):
            r, c = divmod(tap.offset, N)
            strides = [(st // N) * c_ld + st % N for st in tap.strides]
            return TensorAccessPattern([c_rows * c_ld], (r + c_shift) * c_ld + c + c_col0, tap.sizes, strides)

        C_tiles = [_move(tap) for tap in C_tiles]
    flat_workers = [w for row in workers for w in row]
    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = n_tiles_per_core
                rtps[row][col][1] = K // k
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                barriers[row][col].set(1)
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            for _ in range(a_fills):
                                A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                        B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


@iron.jit
def conv_rt(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    mode: CompileTime[str],
    a_rows: CompileTime[int] = 0,
    a_c: CompileTime[int] = 0,
    a_w2: CompileTime[int] = 0,
    a_kh: CompileTime[int] = 0,
    c_rows: CompileTime[int] = 0,
    c_shift: CompileTime[int] = 0,
    c_ld: CompileTime[int] = 0,
    c_col0: CompileTime[int] = 0,
):
    return _build(iron.get_current_device(), M, K, N, m, k, n, n_aie_cols, mode, a_rows, a_c, a_w2, a_kh, c_rows, c_shift,
                  c_ld, c_col0)

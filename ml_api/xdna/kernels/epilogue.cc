// Conv GEMM epilogue for conv_rt: bias + leaky ReLU (+ int8 requantisation)
// on a core's accumulator tile, written in the output dtype. Vectorised: the
// AIE-ML core has no scalar floating-point unit.
//
// The tile is in the matmul kernel's blocked layout [m/r][n/t][r][t]; the
// output keeps that layout (the memtile reorders it on the way out).
//
// Parameter block (prm), laid out by pack.py:
//   bf16 (channels along n, r=4 t=4):
//     float [0] unused, [1] leaky slope, [2 .. 7] unused (32-byte header:
//       16-lane bf16 loads need 32-byte alignment)
//     bfloat16 bias_rep[n/t][16] from byte 32: block i2's bias, one 16-lane
//       vector = bias[i2*4 .. i2*4+3] repeated for the 4 rows
//   int8 (transposed GEMM, channels along m, r=4 t=8):
//     int32 [0] output zero point, [1] shift, [2], [3] unused
//     int32 bias[m] from byte 16: bias in accumulator units
//     int16 mult_pos[m], mult_neg[m]: q = round((acc + bias) * mult >> shift) + zp,
//       mult_neg = mult_pos * leaky slope

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef DIM_M
#error "DIM_M not set"
#endif
#ifndef DIM_N
#error "DIM_N not set"
#endif

extern "C" {

#ifdef EPI_BF16
void epi_bf16(const float *__restrict acc, bfloat16 *__restrict out, const float *__restrict prm) {
  constexpr int r = 4, t = 4, nb = DIM_N / t;
  const bfloat16 alpha = (bfloat16)prm[1];
  const bfloat16 *bias_rep = (const bfloat16 *)(prm + 8);
  const aie::vector<bfloat16, 16> ones = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<bfloat16, 16> va = aie::broadcast<bfloat16, 16>(alpha);
  for (int i0 = 0; i0 < DIM_M / r; i0++)
    for (int i2 = 0; i2 < nb; i2++) {
      const int base = (i0 * nb + i2) * r * t;
      aie::accum<accfloat, 16> y;
      y.from_vector(aie::load_v<16>(acc + base));
      y = aie::mac(y, aie::load_v<16>(bias_rep + i2 * 16), ones);
      aie::vector<bfloat16, 16> v = y.to_vector<bfloat16>();
      aie::vector<bfloat16, 16> neg = aie::mul(v, va).to_vector<bfloat16>();
      aie::store_v(out + base, aie::max(v, neg));
    }
}
#endif

#ifdef EPI_I8
void epi_i8(const int32_t *__restrict acc, int8_t *__restrict out, const float *__restrict prm) {
  constexpr int r = 4, t = 8, nb = DIM_N / t;
  const int32_t *hdr = (const int32_t *)prm;
  const int32_t zp = hdr[0];
  const int shift = hdr[1];
  const int32_t *bias = hdr + 4;
  const int16_t *mpos = (const int16_t *)(bias + DIM_M);
  const int16_t *mneg = mpos + DIM_M;
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  const aie::vector<int32_t, 16> vzp = aie::broadcast<int32_t, 16>(zp);
  const aie::vector<int32_t, 16> zero = aie::zeros<int32_t, 16>();
  for (int i0 = 0; i0 < DIM_M / r; i0++) {
    // the block's 4 channels as two 16-lane halves: rows (ii, ii+1) x 8 patches
    aie::vector<int32_t, 16> vb[2];
    aie::vector<int16_t, 16> vp[2], vn[2];
    for (int h = 0; h < 2; h++) {
      const int c = i0 * r + 2 * h;
      vb[h] = aie::concat(aie::broadcast<int32_t, 8>(bias[c]), aie::broadcast<int32_t, 8>(bias[c + 1]));
      vp[h] = aie::concat(aie::broadcast<int16_t, 8>(mpos[c]), aie::broadcast<int16_t, 8>(mpos[c + 1]));
      vn[h] = aie::concat(aie::broadcast<int16_t, 8>(mneg[c]), aie::broadcast<int16_t, 8>(mneg[c + 1]));
    }
    for (int i2 = 0; i2 < nb; i2++) {
      const int base = (i0 * nb + i2) * r * t;
      for (int h = 0; h < 2; h++) {
        aie::vector<int32_t, 16> y = aie::add(aie::load_v<16>(acc + base + 16 * h), vb[h]);
        aie::vector<int16_t, 16> mult = aie::select(vn[h], vp[h], aie::gt(y, zero));
        aie::accum<acc64, 16> q;
        q.from_vector(vzp, shift);
        q = aie::mac(q, y, mult);
        aie::store_v(out + base + 16 * h, q.to_vector<int8_t>(shift));
      }
    }
  }
}
#endif

} // extern "C"

// Copies the epilogue parameters out of the stream element that carries them
// (the first chunk of each weight panel) into a core-local buffer, so the
// element can be released before the reduction loop starts.

#include <stdint.h>

#ifndef PARAM_FLOATS
#error "PARAM_FLOATS not set"
#endif

extern "C" {

void load_params(const int8_t *__restrict chunk, float *__restrict prm) {
  const float *src = (const float *)chunk;
  for (int i = 0; i < PARAM_FLOATS; i++)
    prm[i] = src[i];
}

} // extern "C"

#ifndef DCT_QUANTIZE_H
#define DCT_QUANTIZE_H

#include <stdint.h>
#include "mjpeg_encoder.h"

#define BLOCK_SIZE_LOG 3
#define BLOCK_SIZE 8
#define BLOCK_AREAL_LOG 6
#define BLOCK_AREAL 64

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

enum
{
  Y_QUANT,
  U_QUANT,
  V_QUANT
};

#define COSUV(i, j, k, l) ((float) (cosv[k][i] * cosv[l][j]))


/* CUDA-stuff... */
#ifdef __cplusplus
extern "C"
{
#endif
  void gpu_dct_quantize(yuv_t *image, dct_t *out);
  void init();
  void cleanup();
#ifdef __cplusplus
}
#endif

#endif /* dct_quantize.h */

/* The AVX-VNNI clients' K-quant and MXFP4 prefill bands: jam_kernels_band8_avx2.c built with one
 * vpdpbusd per dot (-mavxvnni, the jam_avxvnni target). Same repack, tiles and results as the AVX2
 * bands; phase 1 is theirs (jam_band8_quant_avx2). */
#define JAM_BAND8_VNNI 1
#include "jam_kernels_band8_avx2.c"

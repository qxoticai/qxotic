# Precision notes (jam matmul backends)

Findings from the backend parity tests (`JamBackendParityTest`, `VectorKernelTest`). Read these before
tightening any matmul tolerance or chasing a "wrong" result at a specific shape.

## int8-activation paths are not bit-exact vs an F32/double reference

Q8_0/Q4_0/k-quant-weight matmuls on the **int8 paths**, both the Java register-tiled `VectorJAM` kernels
and the native libjam kernels, quantize the **activation to int8** and accumulate in tile order. So they
differ from a sequential double-precision dot (the reference) by roughly the int8 quantization error:

| path | typical relative error vs double ref |
|---|---|
| `ScalarJAM` / `dot()` floor (full F32) | ~1e-5 (float-vs-double rounding) |
| `VectorJAM` tiles / `NativeJAM` (int8 activation) | ~0.4–1% |

The parity tests bound the error **relative to `sumAbs = Σ|wᵢ·aᵢ|`**, NOT to the output value: cancellation
makes `|value| ≪ sumAbs`, so a `|value|`-based tolerance fails spuriously. Concretely:

- `JamBackendParityTest` asserts `|got − ref| ≤ relTol·sumAbs + 1e-3`, with `relTol = 1e-3` for the F32
  `ScalarJAM` floor and `relTol = 1e-2` for the int8 `VectorJAM` / `NativeJAM` paths.
- `VectorKernelTest` asserts `≤ 1e-2·sumAbs + 1e-3`.

These bounds are the *expected* magnitude of int8-activation quantization, not slop: tightening them below
~`1e-2·sumAbs` will fail correct kernels. The F32/F16/BF16 dense paths keep activations in float and stay at
the `~1e-3·sumAbs` floor.

## Activation scale granularity per path

The activation scale is what sets the int8 error, and it differs by weight family and ISA:

| path | activation scale | matches |
|---|---|---|
| 32-block weights (Q8_0/Q4_0/Q5_0/MXFP4), every ISA | one per 32 elements | llama.cpp's Q8_0 activations |
| K-quant weights (Q4_K/Q5_K/Q6_K), AVX-512-VNNI band (`jam_kernels_band32_avx512.c`) | one per 256 elements | llama.cpp's Q8_K activations |
| K-quant weights, AVX2 / AVX-VNNI / SSE3 / ARM | one per 32 elements | (finer than llama.cpp) |

The AVX-512 K-quant band keeps the sub-block scales in the integer domain (`vpmulld`), which needs one activation scale per 256-block; that is exactly llama.cpp's precision model for the K-quants, and the C suite's `RP` (per-256 requant) reference is the one it is measured against.
The band is still bounded by the same `1e-2·sumAbs` tolerance as the per-32 paths.

The band serves the 32-block quants from 3 columns and the K-quants from 8.
The two thresholds differ on purpose: for the 32-block quants the band and the row kernels below it use the same per-32 activation scales, so moving the threshold changes speed only.
For the K-quants the row kernels below the band quantize per 32 and the band per 256, so their threshold decides which calls get which granularity, and it stays where it was.

## Arithmetic error is far below the quantization error

Given the int8 activation codes, what a kernel adds on top is float rounding, and it is seven orders of magnitude smaller than the quantization itself.
Measured against a double-precision dot of the exact weight values and the kernel's own activation codes, relative to `sumAbs`, over every format and a grid of shapes:

| path | rms | worst |
|---|---|---|
| AVX-512 band, K-quants | ~6e-9 | ~4e-8 |
| AVX-512 band, 32-block quants | ~1e-8 | ~1e-7 |
| K-quant row kernels, every width | ~1.4e-8 | ~5e-8, cancelling rows up to 2e-6 |

The AVX-512 band's 32-block dots carry a bias term, 128 times the sum of a row's codes, because the activations are stored unsigned.
It is subtracted as an integer before the conversion to float (the accumulators start from its negative), not as a second float product afterwards: the float then holds the exact integer dot, and the rms error halved.
The 256-bit K-quant row kernels take a row's min term off its running sum block by block, for the same reason: the result is never the small difference of two large sums built up apart.

## History: the seq=7 Q8_0 bug (FIXED)

libjam's Q8_0 gemm was once ~2% wrong at `n == 7` only, a token-remainder edge in the sign-trick weight
repack: `_mm256_sign_epi8` on a `-128` weight byte computes `+128`, which wraps back to `-128` and flips that
term's sign. Fixed in `29982c02` by clamping the repacked weight to `[-127, 127]` (real GGUF Q8_0 never
emits `-128`; the clamp just makes the kernel robust to any input). `n == 7` is now part of the parity matrix.
`JamBackendParityTest` runs every dtype at `n ∈ {7, 8, 13, 16}`, and the native suite sweeps shapes like
`{16,7,64}` / `{104,7,256}`, and both pass. Recorded here so a future `n == 7` oddity isn't re-investigated
from scratch.

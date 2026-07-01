# Precision notes (jam matmul backends)

Findings from the backend parity tests (`JamBackendParityTest`, `VectorKernelTest`). Read these before
tightening any matmul tolerance or chasing a "wrong" result at a specific shape.

## int8-activation paths are not bit-exact vs an F32/double reference

Q8_0/Q4_0/k-quant-weight matmuls on the **int8 paths** — both the Java register-tiled `VectorJAM` kernels
and the native libjam kernels — quantize the **activation to int8** and accumulate in tile order. So they
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

These bounds are the *expected* magnitude of int8-activation quantization, not slop — tightening them below
~`1e-2·sumAbs` will fail correct kernels. The F32/F16/BF16 dense paths keep activations in float and stay at
the `~1e-3·sumAbs` floor.

## History: the seq=7 Q8_0 bug (FIXED)

libjam's Q8_0 gemm was once ~2% wrong at `n == 7` only — a token-remainder edge in the sign-trick weight
repack: `_mm256_sign_epi8` on a `-128` weight byte computes `+128`, which wraps back to `-128` and flips that
term's sign. Fixed in `29982c02` by clamping the repacked weight to `[-127, 127]` (real GGUF Q8_0 never
emits `-128`; the clamp just makes the kernel robust to any input). `n == 7` is now part of the parity matrix
— `JamBackendParityTest` runs every dtype at `n ∈ {7, 8, 13, 16}`, and the native suite sweeps shapes like
`{16,7,64}` / `{104,7,256}` — and passes. Recorded here so a future `n == 7` oddity isn't re-investigated
from scratch.

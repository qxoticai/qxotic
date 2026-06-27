# Precision notes (JAM matmul backends)

Findings from the JAM backend parity tests (`JamBackendTest`) and `KernelParityTest`. Read these before
tightening any matmul tolerance or chasing a "wrong" result at a specific shape.

## int8-activation paths are not bit-exact vs an F32/double reference

Q8_0/Q4_0/…-weight matmuls on the **VNNI paths** — both the Java register-tiled vector kernels and the
native libjam kernel — quantize the **activation to int8** and accumulate in tile order. So they differ
from a sequential double-precision dot (the reference) by roughly the int8 quantization error:

| path | typical relative error vs double ref |
|---|---|
| `ScalarJAM` / `dot()` floor (full F32) | ~1e-5 (float-vs-double rounding) |
| Java vector tiles / native libjam (int8 activation) | ~0.4–1% |

So the parity tests use **`relTol = 1e-3` for the F32 dot floor** and **`1e-2` for the int8 / tiled paths**
(relative to `sumAbs = Σ|wᵢ·aᵢ|`, NOT to the output value — cancellation makes `|value| ≪ sumAbs`, which
is why a `|value|`-based tolerance spuriously fails). This mirrors `KernelParityTest.testGemm` (5e-3).

## Pre-existing correctness bugs at seq/n = 7 (NOT in the JAM backends)

Both surfaced by the parity tests; both independent of `VectorJAM`/`ScalarJAM`/the JAM interface:

1. **libjam Q8_0 gemm at `n == 7` is ~2% wrong.** Only `n == 7`, only Q8_0; `n = 8/13/16` and every other
   dtype are correct. Looks like a token-remainder edge case in the C kernel. `NativeJAM` inherits it.
   Potentially a real inference bug for **7-token prompts** when jam is the active backend — worth a look.
2. **`FloatTensor.gemm` (`w.gemm`) Q8_0 at `seq == 7` is wrong** — this is `KernelParityTest`'s 27 `Q8_0.gemm`
   failures (tile-independent; present with the full make flags). It's in the `w.gemm` *dispatch wrapper*,
   not the kernel: **`vectorGemm512F32` called directly (by `VectorMatMul` and `VectorJAM`) is correct at
   seq=7.** So inference via the Vector path is fine; the bug is in the `w.gemm` entry point.

`JamBackendTest` deliberately omits `n == 7` (it tests the JAM backends, not these pre-existing kernel bugs).

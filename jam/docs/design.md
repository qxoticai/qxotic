# jam design notes

How the native library selects and runs its kernels. For usage see the [README](../README.md); for
building see [BUILDING.md](../BUILDING.md).

## Backends

The instruction sets per architecture is in the [README](../README.md#backends); this is how a context
routes between its executors.

`JAM_ISA=auto` is the default and picks the best available. On Apple Silicon that includes the
Metal backend. Name a CPU rung (`JAM_ISA=i8mm`) to stay CPU-only, or `JAM_ISA=metal` to insist on
it. With Metal active the ctx keeps both executors and routes by measured shape. Dense weights go
to the GPU at every n, since its wider DRAM path wins even at n==1. Quantized weights go to the GPU
only for n>=16, that is prefill, where the block quants run simdgroup-matrix MMA kernels (half
operands, float accumulation, 64x32 tiles), while one-column and small-n decode stays on the CPU
SDOT and I8MM kernels. All Metal calls are zero-copy: W, A and C are borrowed through page-rounded
`newBufferWithBytesNoCopy` views over the caller's unified-memory pages and released after the
synchronous wait, so there are no uploads and no result copies, and strided views are consumed
directly. `JAM_METAL_PROFILE=1` prints per-call encode, submit, wait and GPU averages at context
destroy. SVE, AMX and SME are not yet implemented.

## Packed weights

Decode streams every weight byte per token, and the GGUF block layouts waste bandwidth (unaligned
fp16 scales) or instructions (k-quant bit unpacking) there. jam therefore defines packed in-memory
layouts for Q4_0, Q4_K, Q5_K and Q6_K (per-4-row-group sections, specified next to the dtype tags
in `jam.h`, never a wire format). The contract is caller-packs, jam-reads, one copy:
`jam_pack_size(ctx, dt, m, k)` says whether this ctx's kernels want the layout for a `[m x k]`
weight, and how many bytes it is. The caller produces the bytes once at load, drops the canonical
copy, and passes `wt | JAM_PACKED` to `jam_mm`. Every engine reads that same copy: the 4x1 decode
GEMVs, the 4x4 sdot prefill kernels, and, on Metal and zero-copy via unified memory, the packed MMA
kernels. Values are exactly the canonical dequant. `jam_pack_abi()` guards packers against layout
drift between jam versions.

## Prefill bands (AVX-512-VNNI)

Every quantized prefill (`n >= 8`) on AVX-512-VNNI runs one tile kernel, `jam_kernels_band32_avx512.c`: a band of 32 weight rows is repacked per call into two 16-row vectors (lanes = rows, 4 consecutive k per byte group), and each activation broadcast feeds both vectors, so the dot loop issues one `vpdpbusd` per half broadcast.
That ratio is the design law: on Zen 5 a 512-bit `vpdpbusd` costs 2 of the 4 vector pipe slots per cycle, a memory broadcast half a slot and a 512-bit ALU op 2 slots, so the tile's efficiency is set by the non-dot ops per dot, not by cache bandwidth (k-chunking the band into L1 measured flat).
The K-quants keep their sub-block scales integer (`vpmulld` + `vpaddd` per 32 k, one float conversion per 256-block) against activations quantized per 256 elements, llama.cpp's Q8_K precision; the 32-block quants keep a float scale and a per-32 activation scale.
Tiles are claimed from an atomic counter, so a worker on an SMT sibling or the slower CCD takes fewer of them.
A group short of 16 rows is padded with zero rows and stored under a mask, and a k that is not a multiple of 256 ends in a shorter stretch, so no shape falls back to a scalar tail.
Every other ISA keeps its own bands (the AVX2 8-row bands, the AVX-VNNI 256-bit bands, the ARM tiles).

## Threading

Threads are not a jam setting.
A provider is created with the host's `JAM.Parallel` (`Provider.create(parallel)`) and runs every task on it, its width being the thread budget; in jinfer that is `-Djinfer.threads`.
A backend may run its own threads instead, under one rule: while `mm` runs it uses at most `width()` cores and the host's workers are idle, and when `mm` returns its own workers are idle.
The native library creates no thread inside a JVM; a C host without an executor of its own gets a small pool sized by `jam_config.nthreads` (0 = every online CPU).

A `jam_ctx` is a serial stream: one `mm` at a time. For concurrent matmuls, use one context per
thread.

## Precision

See [PRECISION.md](../PRECISION.md) for the tolerance findings from the backend parity tests.

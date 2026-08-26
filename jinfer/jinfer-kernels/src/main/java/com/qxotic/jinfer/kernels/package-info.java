/**
 * Compute kernels over {@code MemoryView<MemorySegment>}: matmul dispatch, norms, RoPE, attention,
 * MoE routing, conversions.
 *
 * <p><b>Internal to jinfer: published so the dependency graph resolves, with no API or
 * compatibility promise. Depend on jinfer-core, the model artifacts, or the framework adapters
 * instead.</b>
 */
package com.qxotic.jinfer.kernels;

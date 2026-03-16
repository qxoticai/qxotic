/**
 * Mojo backend for AMD GPUs (experimental).
 *
 * <p>This backend provides GPU acceleration using the Mojo programming language:
 *
 * <ul>
 *   <li>Experimental Mojo kernel execution
 *   <li>AMD GPU support via HIP integration
 *   <li>Python-like syntax with systems programming performance
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>Mojo SDK (Modular MAX)
 *   <li>mojo compiler (for runtime kernel compilation)
 *   <li>ROCm 5.0+ (for HIP integration)
 *   <li>AMD GPU (RDNA2/CDNA2 or newer recommended)
 *   <li>Platform: Linux only
 * </ul>
 *
 * <p><b>Output:</b> Mojo kernels compiled to native GPU code via MLIR/LLVM
 *
 * <p><b>Status:</b> Early development - APIs subject to change
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

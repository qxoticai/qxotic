/**
 * HIP/ROCm backend for AMD GPUs.
 *
 * <p>This backend provides GPU acceleration on AMD hardware:
 *
 * <ul>
 *   <li>ROCm/HIP kernel execution
 *   <li>AMD GPU memory management
 *   <li>Optimized for RDNA and CDNA architectures
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>ROCm 5.0+ runtime
 *   <li>hipcc compiler (for runtime kernel compilation)
 *   <li>Platform: Linux only
 * </ul>
 *
 * <p><b>Output:</b> HIP device code compiled to native GPU kernels
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

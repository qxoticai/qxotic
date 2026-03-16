/**
 * CUDA backend for NVIDIA GPUs.
 *
 * <p>This backend provides GPU acceleration on NVIDIA hardware:
 *
 * <ul>
 *   <li>CUDA kernel execution
 *   <li>NVIDIA GPU memory management
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>CUDA Toolkit 11.0+ or 12.x
 *   <li>nvcc compiler (for runtime kernel compilation)
 *   <li>NVIDIA GPU (Compute Capability 5.0+)
 *   <li>Platform: Linux or Windows
 * </ul>
 *
 * <p><b>Output:</b> CUDA device code (.cubin/.ptx) compiled to native GPU kernels
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

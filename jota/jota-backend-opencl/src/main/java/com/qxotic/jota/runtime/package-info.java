/**
 * OpenCL backend for cross-platform GPU acceleration.
 *
 * <p>This backend provides portable GPU execution:
 *
 * <ul>
 *   <li>OpenCL kernel execution
 *   <li>Cross-platform GPU support (NVIDIA, AMD, Intel)
 *   <li>Works with older GPUs and integrated graphics
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>OpenCL 1.2+ runtime and drivers
 *   <li>No external compiler required (kernels compiled at runtime via clBuildProgram)
 *   <li>Platform: Linux, macOS, or Windows
 * </ul>
 *
 * <p><b>Output:</b> OpenCL kernels compiled at runtime via clBuildProgram
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

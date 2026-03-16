/**
 * Metal backend for Apple GPUs.
 *
 * <p>This backend provides GPU acceleration on Apple Silicon:
 *
 * <ul>
 *   <li>Metal kernel execution
 *   <li>Apple GPU memory management
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>macOS 11.0+ (Big Sur or later)
 *   <li>Xcode Command Line Tools (for runtime kernel compilation)
 *   <li>Apple Silicon Mac (M*)
 * </ul>
 *
 * <p><b>Output:</b> Metal Shading Language kernels compiled to .metallib binaries
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

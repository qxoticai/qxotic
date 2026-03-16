/**
 * C native backend for Jota.
 *
 * <p>This backend provides CPU execution via native C code:
 *
 * <ul>
 *   <li>Optimized CPU kernels compiled at runtime
 *   <li>GraalVM Native Image compatible
 *   <li>Fallback backend when no GPU is available
 *   <li>Maximum compatibility across platforms
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>C compiler: gcc or clang (for runtime kernel compilation)
 *   <li>Platform: Linux, macOS, or Windows
 * </ul>
 *
 * <p><b>Output:</b> Native shared library (.so/.dylib/.dll) with compiled C kernels
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

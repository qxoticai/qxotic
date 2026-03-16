/**
 * Panama Foreign Function & Memory API backend for Jota.
 *
 * <p>This backend uses Java's Panama API to provide:
 *
 * <ul>
 *   <li>Native memory access via MemorySegment
 *   <li>Zero-copy data transfer
 *   <li>Runtime kernel compilation to Java bytecode
 * </ul>
 *
 * <p><b>Runtime Requirements:</b>
 *
 * <ul>
 *   <li>Java 25+ with Panama Foreign Function & Memory API (preview)
 *   <li>No external compilers required - uses javax.tools.JavaCompiler
 * </ul>
 *
 * <p><b>Output:</b> Compiled Java bytecode, native memory access via Panama's MemorySegment
 *
 * <p><b>Limitations:</b> Not compatible with GraalVM Native Image (requires JIT)
 *
 * @since 0.1.0
 */
package com.qxotic.jota.runtime;

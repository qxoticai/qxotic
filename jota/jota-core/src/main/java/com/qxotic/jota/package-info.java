/**
 * Core tensor API for the Jota tensor algebra library.
 *
 * <p>This package provides the foundational tensor types and operations including:
 *
 * <ul>
 *   <li>{@link com.qxotic.jota.tensor.Tensor} - The core tensor interface with lazy evaluation
 *   <li>{@link com.qxotic.jota.tensor.TensorOps} - Tensor operations interface
 *   <li>{@link com.qxotic.jota.tensor.Tracer} - Expression tracing for kernel compilation
 *   <li>{@link com.qxotic.jota.Device} - Device abstraction for different backends
 * </ul>
 *
 * <p>Jota supports multiple execution backends including Panama (JVM), C, CUDA, HIP, Metal, and
 * OpenCL.
 *
 * @since 0.1.0
 */
package com.qxotic.jota;

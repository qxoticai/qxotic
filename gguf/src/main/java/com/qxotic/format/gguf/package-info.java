/**
 * Provides the public API for handling GGUF files used to store large language models and their
 * associated metadata.
 *
 * <p>The main interfaces and types of this package are:
 *
 * <ul>
 *   <li>{@link GGUF} - The core interface for reading and managing GGUF files, providing
 *       functionality to:
 *       <ul>
 *         <li>Read and write GGUF files
 *         <li>Access and manage metadata
 *         <li>Handle tensor information
 *         <li>Manage file alignment and version information
 *       </ul>
 *   <li>{@link Builder} - A builder interface for creating and modifying GGUF instances, providing
 *       functionality to:
 *       <ul>
 *         <li>Create new GGUF instances from scratch or existing ones
 *         <li>Modify metadata with type-safe methods
 *         <li>Add, update and remove tensors
 *         <li>Configure version and alignment settings
 *       </ul>
 *   <li>{@link GGMLType} - An enumeration of supported tensor data types, including:
 *       <ul>
 *         <li>Standard types (F32, F16, BF16, I8, I16, I32, I64, F64)
 *         <li>Quantized types (Q4_0, Q4_1, Q8_0, Q8_1, etc.)
 *         <li>K-quantized types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
 *         <li>Block size and memory layout information for each type
 *       </ul>
 * </ul>
 *
 * <p>The package supports various tensor data types and quantization schemes used in GGML-based
 * language models, with utilities for calculating memory layouts and sizes.
 *
 * <p><b>Thread Safety:</b> Instances of {@link GGUF} are immutable and thread-safe for reading.
 * However, {@link Builder} instances are mutable and not thread-safe; callers must synchronize
 * access if sharing across threads.
 *
 * @see GGUF
 * @see Builder
 * @see GGMLType
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 * @see <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>
 */
package com.qxotic.format.gguf;

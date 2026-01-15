/**
 * Provides the public API for handling GGUF files used to store large language models
 * and their associated metadata.
 * <p>
 * The main interfaces and types of this package are:
 * <ul>
 *   <li>{@link .format.gguf.GGUF} - The core interface for reading and managing GGUF files,
 *       providing functionality to:
 *       <ul>
 *         <li>Read and write GGUF files</li>
 *         <li>Access and manage metadata</li>
 *         <li>Handle tensor information</li>
 *         <li>Manage file alignment and version information</li>
 *       </ul>
 *   </li>
 *   <li>{@link .format.gguf.Builder} - A builder interface for creating and modifying GGUF instances,
 *       providing functionality to:
 *       <ul>
 *         <li>Create new GGUF instances from scratch or existing ones</li>
 *         <li>Modify metadata with type-safe methods</li>
 *         <li>Add, update and remove tensors</li>
 *         <li>Configure version and alignment settings</li>
 *       </ul>
 *   </li>
 *   <li>{@link .format.gguf.GGMLType} - An enumeration of supported tensor data types, including:
 *       <ul>
 *         <li>Standard types (F32, F16, BF16, I8, I16, I32, I64, F64)</li>
 *         <li>Quantized types (Q4_0, Q4_1, Q8_0, Q8_1, etc.)</li>
 *         <li>K-quantized types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)</li>
 *         <li>Block size and memory layout information for each type</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * <p>
 * The package supports various tensor data types and quantization schemes used in GGML-based
 * language models, with utilities for calculating memory layouts and sizes.
 *
 * @see .format.gguf.GGUF
 * @see .format.gguf.Builder
 * @see .format.gguf.GGMLType
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format specification</a>
 * @see <a href="https://github.com/ggerganov/llama.cpp">llama.cpp</a>
 */
package ai.qxotic.format.gguf;

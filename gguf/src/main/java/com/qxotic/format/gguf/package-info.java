/**
 * Public API for reading and writing GGUF headers, which describe the metadata and tensors in
 * GGML-based model files. Tensor payload bytes are outside this API's scope.
 *
 * <p>{@link GGUF} is the entry point for reading and writing; {@link Builder} creates new instances
 * or modifies copies of existing ones; {@link TensorEntry} describes a tensor and {@link GGMLType}
 * enumerates the tensor data types.
 *
 * <p>{@link GGUF} header structure is read-only and may be shared when its array-valued metadata is
 * not mutated. Array values are returned by reference and may also be shared with a {@link
 * Builder}; builders are mutable and not thread-safe.
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 * @see <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>
 */
package com.qxotic.format.gguf;

/**
 * Tokenization library for LLMs.
 *
 * <p>This package provides text tokenization for large language models:
 *
 * <ul>
 *   <li>IntSequence-first tokenizer API via {@link com.qxotic.toknroll.Tokenizer}
 *   <li>Tiktoken BPE tokenizer support
 *   <li>High-level tokenizer factories and lossy pipeline wrappers via {@link
 *       com.qxotic.toknroll.Toknroll}
 *   <li>Byte-level symbol utility via {@link com.qxotic.toknroll.ByteLevel}
 *   <li>Vocabulary management
 *   <li>Compatibility with OpenAI, Llama, Mistral, and other models
 *   <li>Matches Python reference implementations
 * </ul>
 *
 * @since 0.1.0
 */
package com.qxotic.toknroll;

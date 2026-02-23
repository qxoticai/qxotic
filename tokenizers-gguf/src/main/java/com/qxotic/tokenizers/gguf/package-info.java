/**
 * Tokenizer support for GGUF models.
 *
 * <p>This module provides tokenizer loading from GGUF files. GGUF supports embedded vocabularies
 * with metadata describing the tokenization scheme.
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // Simple usage - one line
 * Tokenizer tokenizer = GGUFTokenizers.fromFile("/path/to/model.gguf");
 *
 * // Advanced - customize registries
 * Tokenizer tokenizer = GGUFTokenizers.fromFile(
 *     path,
 *     GGUFTokenizers.preTokenizers(),
 *     GGUFTokenizers.tokenizers()
 * );
 * }</pre>
 *
 * <h2>GGUF Tokenizer Metadata</h2>
 *
 * <p>GGUF uses the following keys to describe the tokenizer:
 *
 * <h3>Tokenizer Model</h3>
 *
 * <ul>
 *   <li><b>tokenizer.ggml.model</b>: The name of the tokenizer model
 *       <ul>
 *         <li>{@code gpt2}: GPT-2 / GPT-NeoX style BPE
 *         <li>{@code llama}: Llama style SentencePiece
 *         <li>{@code replit}: Replit style SentencePiece
 *         <li>{@code rwkv}: RWKV tokenizer
 *       </ul>
 *   <li><b>tokenizer.ggml.tokens</b>: List of tokens indexed by token ID
 *   <li><b>tokenizer.ggml.scores</b>: Score/probability of each token (optional)
 *   <li><b>tokenizer.ggml.token_type</b>: Token type (1=normal, 2=unknown, 3=control, 4=user
 *       defined, 5=unused, 6=byte)
 *   <li><b>tokenizer.ggml.merges</b>: BPE merges (optional, for BPE tokenizers)
 *   <li><b>tokenizer.ggml.added_tokens</b>: Tokens added after training (optional)
 * </ul>
 *
 * <h3>Special Tokens</h3>
 *
 * <ul>
 *   <li><b>tokenizer.ggml.bos_token_id</b>: Beginning of sequence marker
 *   <li><b>tokenizer.ggml.eos_token_id</b>: End of sequence marker
 *   <li><b>tokenizer.ggml.unknown_token_id</b>: Unknown token
 *   <li><b>tokenizer.ggml.separator_token_id</b>: Separator token
 *   <li><b>tokenizer.ggml.padding_token_id</b>: Padding token
 * </ul>
 *
 * <h3>Pre-tokenizer</h3>
 *
 * <ul>
 *   <li><b>tokenizer.ggml.pre</b>: Pre-tokenizer name (e.g., {@code llama}, {@code qwen2}, {@code
 *       smollm}, {@code tekken})
 * </ul>
 *
 * <h2>Official Specification</h2>
 *
 * <p>For the complete GGUF specification, see: <a
 * href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#tokenizer">
 * https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#tokenizer</a>
 *
 * @see com.qxotic.tokenizers.gguf.GGUFTokenizers
 * @see com.qxotic.tokenizers.gguf.GGUFPreTokenizerRegistry
 * @see com.qxotic.tokenizers.gguf.GGUFTokenizerRegistry
 */
package com.qxotic.tokenizers.gguf;

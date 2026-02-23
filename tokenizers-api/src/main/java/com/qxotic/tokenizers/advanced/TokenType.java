package com.qxotic.tokenizers.advanced;

/**
 * A marker interface that defines token classification types used in tokenization systems. This
 * interface serves as a base for different token type classification schemes, allowing for
 * extensible token categorization across different tokenizer implementations and model formats.
 *
 * <p>Token types are used to:
 *
 * <ul>
 *   <li>Classify tokens based on their role (e.g., normal text, control tokens, special markers)
 *   <li>Determine token processing behavior during tokenization and detokenization
 *   <li>Support different token classification schemes for various model formats
 * </ul>
 *
 * <p>Common implementations include:
 *
 * <pre>{@code
 * // Standard GGUF token types
 * public enum StandardTokenType implements TokenType {
 *     NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE
 * }
 *
 * // Custom token types for specific models
 * public enum CustomTokenType implements TokenType {
 *     TEXT, SEPARATOR, SPECIAL, METADATA
 * }
 * }</pre>
 *
 * @see StandardTokenType
 */
public interface TokenType {}

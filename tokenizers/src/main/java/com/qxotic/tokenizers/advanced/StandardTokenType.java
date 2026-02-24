package com.qxotic.tokenizers.advanced;

/**
 * Standard token types as defined by the GGUF specification in <a
 * href="https://github.com/ggerganov/llama.cpp">llama.cpp</a>. These types categorize different
 * kinds of tokens in the model's vocabulary and determine how they should be processed during
 * tokenization and text generation.
 *
 * <p>The numeric IDs correspond directly to the token type values in the GGUF format specification.
 * This enum implements {@link TokenType} to provide a standardized way to identify token
 * categories.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF Format
 *     Specification</a>
 */
public enum StandardTokenType implements TokenType {
    /**
     * Regular vocabulary token (id=1). Represents normal text tokens that are part of the model's
     * primary vocabulary.
     */
    NORMAL(1),

    /**
     * Unknown or special token (id=2). Used for tokens that don't correspond to known vocabulary
     * items, often represented as &lt;unk&gt; in the model's output.
     */
    UNKNOWN(2),

    /**
     * Control token (id=3). Special tokens that control model behavior or text generation, such as
     * beginning-of-text (BOT), end-of-text (EOT), or padding tokens.
     */
    CONTROL(3),

    /**
     * User-defined token (id=4). Custom tokens added to the vocabulary for specific use cases or
     * fine-tuning scenarios.
     */
    USER_DEFINED(4),

    /** Unused token type (id=5). Reserved for future use in the GGUF specification. */
    UNUSED(5),

    /**
     * Byte token (id=6). Represents raw byte values, typically used in byte-pair encoding (BPE) or
     * for handling unknown Unicode characters.
     */
    BYTE(6);

    /** The numeric identifier of this token type as defined in the GGUF specification. */
    private final int id;

    StandardTokenType(int id) {
        this.id = id;
    }

    /**
     * Returns the numeric identifier for this token type as defined in the GGUF specification.
     *
     * @return the GGUF identifier for this token type
     */
    public int getId() {
        return id;
    }
}

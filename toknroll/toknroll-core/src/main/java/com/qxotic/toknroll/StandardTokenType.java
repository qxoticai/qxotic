package com.qxotic.toknroll;

/**
 * Token types defined by the GGUF specification (llama.cpp). The {@linkplain #getId() numeric IDs}
 * match the token type values in the GGUF format.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF Format
 *     Specification</a>
 */
public enum StandardTokenType implements TokenType {
    /** Regular text token (GGUF id 1). */
    NORMAL(1),

    /** Unknown token, often rendered as {@code <unk>} (GGUF id 2). */
    UNKNOWN(2),

    /** Control token such as beginning/end-of-text or padding (GGUF id 3). */
    CONTROL(3),

    /** Custom token added for specific use cases or fine-tuning (GGUF id 4). */
    USER_DEFINED(4),

    /** Reserved by the GGUF specification (GGUF id 5). */
    UNUSED(5),

    /** Raw byte value token (GGUF id 6). */
    BYTE(6);

    /** The numeric identifier of this token type as defined in the GGUF specification. */
    private final int id;

    StandardTokenType(int id) {
        this.id = id;
    }

    /**
     * Returns the GGUF numeric identifier of this token type.
     *
     * @return GGUF token type id
     */
    public int getId() {
        return id;
    }
}

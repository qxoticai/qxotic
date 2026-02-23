package com.qxotic.tokenizers.gguf;

public class GGUFTokenizerException extends RuntimeException {

    public GGUFTokenizerException(String message) {
        super(message);
    }

    public GGUFTokenizerException(String message, Throwable cause) {
        super(message, cause);
    }
}

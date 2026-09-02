package com.qxotic.format.gguf;

/**
 * Unchecked exception thrown when GGUF content violates the format specification, e.g. bad magic
 * number, unsupported version, or invalid metadata types.
 */
public class GGUFFormatException extends RuntimeException {

    public GGUFFormatException(String message) {
        super(message);
    }

    public GGUFFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}

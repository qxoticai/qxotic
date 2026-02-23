package com.qxotic.tokenizers.hf;

/**
 * Exception thrown when loading a HuggingFace tokenizer fails.
 *
 * <p>This exception provides detailed context about what went wrong during tokenizer loading,
 * including the file path and specific validation error.
 */
public class HuggingFaceTokenizerException extends RuntimeException {

    /** Path to the file that caused the error, if applicable. */
    private final String filePath;

    /**
     * Creates a new exception with a message.
     *
     * @param message the error message
     */
    public HuggingFaceTokenizerException(String message) {
        super(message);
        this.filePath = null;
    }

    /**
     * Creates a new exception with a message and file path.
     *
     * @param filePath the path to the file that caused the error
     * @param message the error message
     */
    public HuggingFaceTokenizerException(String filePath, String message) {
        super(message);
        this.filePath = filePath;
    }

    /**
     * Creates a new exception with a message and cause.
     *
     * @param message the error message
     * @param cause the underlying cause
     */
    public HuggingFaceTokenizerException(String message, Throwable cause) {
        super(message, cause);
        this.filePath = null;
    }

    /**
     * Creates a new exception with a message, file path, and cause.
     *
     * @param filePath the path to the file that caused the error
     * @param message the error message
     * @param cause the underlying cause
     */
    public HuggingFaceTokenizerException(String filePath, String message, Throwable cause) {
        super(message, cause);
        this.filePath = filePath;
    }

    /**
     * Returns the path to the file that caused the error, or null if not applicable.
     *
     * @return the file path or null
     */
    public String getFilePath() {
        return filePath;
    }
}

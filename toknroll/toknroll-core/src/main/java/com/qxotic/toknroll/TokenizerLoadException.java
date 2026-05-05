package com.qxotic.toknroll;

/** Unchecked exception raised when tokenizer loading fails due to I/O or remote fetch errors. */
public final class TokenizerLoadException extends RuntimeException {
    /**
     * @param message human-readable error description
     */
    public TokenizerLoadException(String message) {
        super(message);
    }

    /**
     * @param message human-readable error description
     * @param cause underlying I/O or network error
     */
    public TokenizerLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}

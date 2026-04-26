package com.qxotic.toknroll.hf;

/** Unchecked exception raised when tokenizer loading fails due to I/O or remote fetch errors. */
public final class TokenizerLoadException extends RuntimeException {
    public TokenizerLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}

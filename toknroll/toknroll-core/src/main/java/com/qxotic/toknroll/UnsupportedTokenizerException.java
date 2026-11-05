package com.qxotic.toknroll;

/**
 * A well-formed tokenizer asking for what toknroll does not implement: another model type, or a
 * normalizer, pre-tokenizer or decoder step outside the supported set. A malformed file is a plain
 * {@link IllegalArgumentException} instead.
 */
public class UnsupportedTokenizerException extends IllegalArgumentException {
    public UnsupportedTokenizerException(String message) {
        super(message);
    }
}

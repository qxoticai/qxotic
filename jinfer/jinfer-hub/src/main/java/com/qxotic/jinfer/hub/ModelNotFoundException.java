package com.qxotic.jinfer.hub;

/**
 * A source names no model that exists or can be reached: no such local file, repository (or one
 * gated or private), file or quant in it, or, offline, no cached copy. An {@link
 * IllegalArgumentException}, as the JDK's {@code UnsupportedCharsetException} is: the argument
 * names nothing. The cause, when there is one, is the hub's answer.
 */
public final class ModelNotFoundException extends IllegalArgumentException {
    private final String source;

    ModelNotFoundException(String source, String message, Throwable cause) {
        super(message, cause);
        this.source = source;
    }

    /** The path or ref that names nothing, as the store read it. */
    public String source() {
        return source;
    }
}

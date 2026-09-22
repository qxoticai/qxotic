package com.qxotic.jinfer.testkit;

import java.util.Objects;

/**
 * A system property overridden for a try-with-resources block, and restored when the block exits:
 * to its previous value, or cleared if it had none.
 *
 * <pre>{@code
 * try (var windows = SystemProperty.override("jinfer.parakeet.chunkSeconds", "10")) {
 *     // the property is "10" here
 * }
 * }</pre>
 */
public final class SystemProperty implements AutoCloseable {

    private final String key;
    private final String previous; // null when the property was not set

    private SystemProperty(String key, String previous) {
        this.key = key;
        this.previous = previous;
    }

    /** Sets {@code key} to {@code value}, or clears it when {@code value} is null, until closed. */
    public static SystemProperty override(String key, String value) {
        Objects.requireNonNull(key, "key");
        String previous =
                value == null ? System.clearProperty(key) : System.setProperty(key, value);
        return new SystemProperty(key, previous);
    }

    @Override
    public void close() {
        if (previous == null) System.clearProperty(key);
        else System.setProperty(key, previous);
    }
}

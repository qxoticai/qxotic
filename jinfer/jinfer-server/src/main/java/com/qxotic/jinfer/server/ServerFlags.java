// The jinfer.server* properties read at run time (works with -D on the JVM and on a native binary).
package com.qxotic.jinfer.server;

import java.util.concurrent.TimeUnit;

/**
 * Every tunable {@code jinfer.server*} system property, in one place - the HTTP layer's counterpart
 * to the engine's {@code RuntimeFlags}. Like that class this is initialized at RUN time even in a
 * native image ({@code --initialize-at-run-time}, see jinfer-parent's native-image buildArgs), so
 * {@code -Djinfer.serverThreads=N} behaves identically on the JVM and on a compiled binary. Adding
 * a flag here without that build arg would silently bake its default into the image.
 */
public final class ServerFlags {
    private ServerFlags() {}

    public static final int SERVER_THREADS = Integer.getInteger("jinfer.serverThreads", 16);
    public static final int SERVER_QUEUE = Integer.getInteger("jinfer.serverQueue", 4);
    public static final long SERVER_MAX_BODY_BYTES =
            Math.min(Long.getLong("jinfer.serverMaxBodyMB", 32), 1024) << 20;
    public static final long SERVER_WRITE_STALL_NANOS =
            TimeUnit.SECONDS.toNanos(Long.getLong("jinfer.serverWriteTimeout", 30));
    public static final int SERVER_MAX_TOKENS =
            Integer.getInteger("jinfer.serverMaxTokens", 4096); // 0 = no completion-token ceiling
    public static final long SERVER_REQUEST_TIMEOUT_NANOS =
            TimeUnit.SECONDS.toNanos(
                    Long.getLong("jinfer.serverRequestTimeout", 300)); // 0 = no generation deadline
}

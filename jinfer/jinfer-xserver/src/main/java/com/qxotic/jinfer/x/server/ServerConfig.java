package com.qxotic.jinfer.x.server;

import com.qxotic.jinfer.x.llm.Sampling;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.Set;

/** Immutable transport and request defaults for one server instance. */
public record ServerConfig(
        InetSocketAddress bind, Defaults defaults, Limits limits, Access access) {

    public ServerConfig {
        if (bind == null) throw new IllegalArgumentException("bind is required");
        if (defaults == null) throw new IllegalArgumentException("defaults are required");
        if (limits == null) throw new IllegalArgumentException("limits are required");
        if (access == null) throw new IllegalArgumentException("access is required");
    }

    /** Safe local defaults; port 0 asks the OS for an ephemeral port. */
    public static ServerConfig local(int port) {
        return new ServerConfig(
                new InetSocketAddress(InetAddress.getLoopbackAddress(), port),
                Defaults.DEFAULTS,
                Limits.DEFAULTS,
                Access.LOCAL);
    }

    /** Request values clients may override. Null sampling means the model's own defaults. */
    public record Defaults(
            Sampling sampling, int maxOutputTokens, boolean think, boolean rawPrompt) {
        public static final Defaults DEFAULTS = new Defaults(null, -1, true, false);

        public Defaults {
            if (maxOutputTokens < -1)
                throw new IllegalArgumentException("maxOutputTokens " + maxOutputTokens);
        }
    }

    /** Resource and protocol limits clients cannot lift. */
    public record Limits(
            int threads,
            int queueCapacity,
            long maxBodyBytes,
            boolean grammar,
            Duration writeTimeout,
            Duration requestTimeout,
            Duration shutdownTimeout) {
        public static final Limits DEFAULTS =
                new Limits(
                        16,
                        4,
                        32L << 20,
                        true,
                        Duration.ofSeconds(30),
                        Duration.ofSeconds(300),
                        Duration.ofSeconds(30));

        public Limits {
            if (threads < 1) throw new IllegalArgumentException("threads " + threads);
            if (queueCapacity < 0)
                throw new IllegalArgumentException("queueCapacity " + queueCapacity);
            if (maxBodyBytes < 1)
                throw new IllegalArgumentException("maxBodyBytes " + maxBodyBytes);
            requirePositive(writeTimeout, "writeTimeout");
            requireNonNegative(requestTimeout, "requestTimeout");
            requireNonNegative(shutdownTimeout, "shutdownTimeout");
        }

        int retryAfterSeconds() {
            return Math.max(1, 2 * (queueCapacity + 1));
        }

        private static void requirePositive(Duration value, String name) {
            if (value == null || value.isNegative() || value.isZero()) {
                throw new IllegalArgumentException(name + " " + value);
            }
        }

        private static void requireNonNegative(Duration value, String name) {
            if (value == null || value.isNegative()) {
                throw new IllegalArgumentException(name + " " + value);
            }
        }
    }

    /** Optional bearer authentication and the exact browser origins allowed by CORS. */
    public record Access(String bearerToken, Set<String> allowedOrigins) {
        public static final Access LOCAL = new Access(null, Set.of("*"));

        public Access {
            if (bearerToken != null && bearerToken.isBlank()) bearerToken = null;
            allowedOrigins = allowedOrigins == null ? Set.of() : Set.copyOf(allowedOrigins);
        }
    }
}

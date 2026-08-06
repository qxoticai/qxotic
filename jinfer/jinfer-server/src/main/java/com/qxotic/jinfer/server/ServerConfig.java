package com.qxotic.jinfer.server;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Sampling;
import java.net.InetSocketAddress;
import java.time.Duration;

/**
 * Everything {@link Server#start} needs, and nothing else. This module reads no system properties
 * and no environment: configuration ARRIVES, so a caller can run two servers with different
 * settings, and so the values a request sees are the values someone passed rather than whatever the
 * JVM was started with. Turning flags, environment and model defaults into one of these is the
 * caller's job - see {@code Main} for the CLI's answer.
 *
 * <p>The grouping is the contract, not decoration: {@link Defaults} is what a REQUEST MAY OVERRIDE
 * and {@link Limits} is what it may not. A knob that moved between the two would be changing what
 * clients are allowed to do, which is exactly the decision that should be hard to make by accident.
 *
 * @param modelName what {@code /v1/models} advertises, what a request's {@code model} must match,
 *     and the prompt cache's identity. A NAME, not a path: nothing here opens the file, and a
 *     caller who built the model in memory still has to be able to say what it is called
 * @param bind the listening address; port 0 binds an ephemeral port, readable from {@link
 *     Server.Running#address}
 * @param cache the prompt cache's own record ({@code blockBudgetBytes} 0 = blocks off, {@code
 *     catalog} null = RAM only)
 */
public record ServerConfig(
        String modelName,
        InetSocketAddress bind,
        Defaults defaults,
        Limits limits,
        PromptCache.Options cache) {

    public ServerConfig {
        if (modelName == null || modelName.isBlank()) {
            throw new IllegalArgumentException("modelName is required");
        }
        if (bind == null) throw new IllegalArgumentException("bind is required");
        if (defaults == null) throw new IllegalArgumentException("defaults are required");
        if (limits == null) throw new IllegalArgumentException("limits are required");
        if (cache == null) throw new IllegalArgumentException("cache options are required");
    }

    /**
     * What a request MAY override, and the value it gets when it says nothing.
     *
     * @param maxTokens the completion budget for a request without {@code max_tokens}; -1 = the
     *     model's own maximum. Not a ceiling - see {@link Limits#maxTokens}
     * @param think whether reasoning runs when a request does not say
     * @param rawPrompt {@code /v1/completions} encodes the prompt special-token aware, so a caller
     *     can place markers itself
     */
    public record Defaults(Sampling sampling, int maxTokens, boolean think, boolean rawPrompt) {

        public Defaults {
            if (sampling == null) throw new IllegalArgumentException("sampling is required");
            if (maxTokens < -1) throw new IllegalArgumentException("maxTokens " + maxTokens);
        }
    }

    /**
     * Ceilings and refusals a request cannot lift. These were {@code jinfer.server*} system
     * properties, read statically from inside the transport - which meant they were per JVM, not
     * per server, and (being read at class initialization) had to be listed in the native image's
     * initialize-at-run-time set or the build machine's values were baked into the binary.
     *
     * @param threads the HTTP handler pool; handlers only parse and block on the worker, so a fixed
     *     pool caps what slow-loris connections can pin
     * @param queueDepth generation requests queued behind the one worker; 0 = reject unless idle
     * @param maxTokens the most any single request may generate, whatever it asks for; 0 = no
     *     ceiling
     * @param grammar whether grammar and {@code response_format} requests are ACCEPTED. False
     *     refuses them with a 400; it must never mean "quietly generate unconstrained"
     * @param writeTimeout how long a streaming write may block before the client is disconnected
     *     (the JDK server has no write timeout, and one dead client would wedge the worker)
     * @param requestTimeout the generation deadline; zero = none
     */
    public record Limits(
            int threads,
            int queueDepth,
            long maxBodyBytes,
            int maxTokens,
            boolean grammar,
            Duration writeTimeout,
            Duration requestTimeout) {

        /** What the {@code jinfer.server*} properties defaulted to. */
        public static final Limits DEFAULTS =
                new Limits(
                        16,
                        4,
                        32L << 20,
                        4096,
                        true,
                        Duration.ofSeconds(30),
                        Duration.ofSeconds(300));

        public Limits {
            if (threads < 1) throw new IllegalArgumentException("threads " + threads);
            if (queueDepth < 0) throw new IllegalArgumentException("queueDepth " + queueDepth);
            if (maxBodyBytes < 1)
                throw new IllegalArgumentException("maxBodyBytes " + maxBodyBytes);
            if (maxTokens < 0) throw new IllegalArgumentException("maxTokens " + maxTokens);
            if (writeTimeout == null || writeTimeout.isNegative() || writeTimeout.isZero()) {
                throw new IllegalArgumentException("writeTimeout " + writeTimeout);
            }
            if (requestTimeout == null || requestTimeout.isNegative()) {
                throw new IllegalArgumentException("requestTimeout " + requestTimeout);
            }
        }

        /** Retry-After seconds suggested when the queue is full. */
        int retryAfterSeconds() {
            return Math.max(1, 2 * (queueDepth + 1));
        }

        // one wither per knob: the CLI sets these one flag at a time, and a 7-argument positional
        // constructor at each -> a transposed threads/queueDepth pair that still compiles
        public Limits withThreads(int threads) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        public Limits withQueueDepth(int queueDepth) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        public Limits withMaxBodyBytes(long maxBodyBytes) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        public Limits withMaxTokens(int maxTokens) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        /** These limits with grammar requests allowed or refused. */
        public Limits withGrammar(boolean grammar) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        public Limits withWriteTimeout(Duration writeTimeout) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }

        public Limits withRequestTimeout(Duration requestTimeout) {
            return new Limits(
                    threads,
                    queueDepth,
                    maxBodyBytes,
                    maxTokens,
                    grammar,
                    writeTimeout,
                    requestTimeout);
        }
    }
}

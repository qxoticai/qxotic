package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import dev.langchain4j.model.output.TokenUsage;
import java.util.Locale;
import java.util.Objects;

/**
 * jinfer's usage accounting: the standard counts plus the prompt-cache read and the phase timings,
 * on every response - so "is it the model or my code" is answerable from the response, without a
 * profiler. Cache read follows the OpenAI-provider pattern ({@code cached_tokens}); the timings
 * mirror spring-ai's {@code JinferUsage}, same names, same nanos.
 *
 * <p>{@link #cachedInputTokens()} is the ground truth: input tokens whose KV was RESTORED rather
 * than computed, whatever served them - a cached-prompt view's prefix, a {@code cachedSessions}
 * extension, or the block layer's best-effort reuse. Healthy traffic on a cached-prompt view
 * restores nearly everything but the newest turn; 0 on a view means the prefill was paid in full (a
 * tools override does this - a one-time stderr warning names it). {@link #servedFrom()} names the
 * source; note a partial restore still reports {@code BLOCKS}, so read the count, not the tier, for
 * how much was saved.
 *
 * <p>{@link #promptNanos()} is the prefill wall time (the engine half of time-to-first-token);
 * {@link #predictedNanos()} the decode wall time - tokens per second is {@code outputTokenCount /
 * predictedNanos} away. {@link #toString()} renders the derived numbers, so a logged usage object
 * IS the diagnosis line.
 */
public final class JinferTokenUsage extends TokenUsage {

    private final int cachedInputTokens;
    private final PromptCache.Tier servedFrom;
    private final long promptNanos;
    private final long predictedNanos;

    /** Built from the engine's own completion - the one construction that cannot transpose. */
    JinferTokenUsage(int promptTokens, ChatEngine.Completion done) {
        this(
                promptTokens,
                done.result().completionTokens(),
                done.restoredTokens(),
                done.tier(),
                done.result().promptNanos(),
                done.result().predictedNanos());
    }

    JinferTokenUsage(
            int inputTokenCount,
            int outputTokenCount,
            int cachedInputTokens,
            PromptCache.Tier servedFrom,
            long promptNanos,
            long predictedNanos) {
        super(inputTokenCount, outputTokenCount);
        this.cachedInputTokens = cachedInputTokens;
        this.servedFrom = servedFrom;
        this.promptNanos = promptNanos;
        this.predictedNanos = predictedNanos;
    }

    /**
     * Input tokens served from cache (KV restored, not computed); part of {@code inputTokenCount}.
     * Sums across {@link #add}, like the standard counts.
     */
    public int cachedInputTokens() {
        return cachedInputTokens;
    }

    /**
     * Which cache tier served the prompt; {@code FRESH} means nothing matched. Null on an aggregate
     * of multiple requests (an AiServices tool loop sums its calls' usages) - a sum has no single
     * serving.
     */
    public PromptCache.Tier servedFrom() {
        return servedFrom;
    }

    /** Prefill wall time in nanos. Sums across {@link #add}: total prefill of the interaction. */
    public long promptNanos() {
        return promptNanos;
    }

    /** Decode wall time in nanos. Sums across {@link #add}: total decode of the interaction. */
    public long predictedNanos() {
        return predictedNanos;
    }

    /**
     * A TokenUsage subclass MUST override add: the superclass double-dispatches to the
     * subclass-typed side, so inheriting it makes two jinfer usages recurse forever. Counts, cache
     * reads and timings all sum (one serial pipeline: summed wall time is the interaction's total);
     * the tier does not survive aggregation.
     */
    @Override
    public JinferTokenUsage add(TokenUsage that) {
        if (that == null) return this;
        JinferTokenUsage j = that instanceof JinferTokenUsage typed ? typed : null;
        return new JinferTokenUsage(
                sum(inputTokenCount(), that.inputTokenCount()),
                sum(outputTokenCount(), that.outputTokenCount()),
                cachedInputTokens + (j == null ? 0 : j.cachedInputTokens),
                null,
                promptNanos + (j == null ? 0 : j.promptNanos),
                predictedNanos + (j == null ? 0 : j.predictedNanos));
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JinferTokenUsage that
                && super.equals(that)
                && cachedInputTokens == that.cachedInputTokens
                && servedFrom == that.servedFrom
                && promptNanos == that.promptNanos
                && predictedNanos == that.predictedNanos;
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                super.hashCode(), cachedInputTokens, servedFrom, promptNanos, predictedNanos);
    }

    @Override
    public String toString() {
        StringBuilder sb =
                new StringBuilder("JinferTokenUsage{input=")
                        .append(inputTokenCount())
                        .append(" (cached=")
                        .append(cachedInputTokens)
                        .append(servedFrom == null ? "" : ", " + servedFrom)
                        .append("), output=")
                        .append(outputTokenCount())
                        .append(String.format(Locale.ROOT, ", prompt=%.2fs", promptNanos / 1e9));
        if (predictedNanos > 0) {
            sb.append(
                    String.format(
                            Locale.ROOT,
                            ", decode=%.1f tok/s",
                            outputTokenCount() * 1e9 / predictedNanos));
        }
        return sb.append('}').toString();
    }
}

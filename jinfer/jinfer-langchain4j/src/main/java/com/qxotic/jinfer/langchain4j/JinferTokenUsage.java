package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.cache.PromptCache;
import dev.langchain4j.model.output.TokenUsage;
import java.util.Objects;

/**
 * jinfer's usage accounting: the standard counts plus the prompt-cache read, on every response -
 * the OpenAI-provider pattern ({@code cached_tokens}), so cache behavior is diagnosable from the
 * response instead of guessed from latency.
 *
 * <p>{@link #cachedInputTokens()} is the ground truth: input tokens whose KV was RESTORED rather
 * than computed, whatever served them - a cached-prompt view's prefix, a {@code cachedSessions}
 * extension, or the block layer's best-effort reuse. Healthy traffic on a cached-prompt view
 * restores nearly everything but the newest turn; 0 on a view means the prefill was paid in full (a
 * tools override does this - a one-time stderr warning names it). {@link #servedFrom()} names the
 * source; note a partial restore still reports {@code BLOCKS}, so read the count, not the tier, for
 * how much was saved.
 */
public final class JinferTokenUsage extends TokenUsage {

    private final int cachedInputTokens;
    private final PromptCache.Tier servedFrom;

    public JinferTokenUsage(
            int inputTokenCount,
            int outputTokenCount,
            int cachedInputTokens,
            PromptCache.Tier servedFrom) {
        super(inputTokenCount, outputTokenCount);
        this.cachedInputTokens = cachedInputTokens;
        this.servedFrom = servedFrom;
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

    /**
     * A TokenUsage subclass MUST override add: the superclass double-dispatches to the
     * subclass-typed side, so inheriting it makes two jinfer usages recurse forever.
     */
    @Override
    public JinferTokenUsage add(TokenUsage that) {
        if (that == null) return this;
        return new JinferTokenUsage(
                sum(inputTokenCount(), that.inputTokenCount()),
                sum(outputTokenCount(), that.outputTokenCount()),
                cachedInputTokens + (that instanceof JinferTokenUsage j ? j.cachedInputTokens : 0),
                null);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JinferTokenUsage that
                && super.equals(that)
                && cachedInputTokens == that.cachedInputTokens
                && servedFrom == that.servedFrom;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), cachedInputTokens, servedFrom);
    }

    @Override
    public String toString() {
        return "JinferTokenUsage{input="
                + inputTokenCount()
                + ", output="
                + outputTokenCount()
                + ", cachedInput="
                + cachedInputTokens
                + ", servedFrom="
                + servedFrom
                + "}";
    }
}

package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.cache.PromptCache;
import dev.langchain4j.model.output.TokenUsage;
import org.junit.jupiter.api.Test;

/**
 * The usage arithmetic, model-free: {@link JinferTokenUsage#add} must sum counts, cache reads and
 * timings (and null the tier - a sum has no single serving), and {@link
 * JinferTokenUsage#toString()} must render the derived numbers, because a logged usage object is
 * the diagnosis line.
 */
class JinferTokenUsageTest {

    @Test
    void addSumsEverythingAndDropsTheTier() {
        JinferTokenUsage a =
                new JinferTokenUsage(100, 10, 80, PromptCache.Tier.BLOCKS, 1_000, 2_000);
        JinferTokenUsage b = new JinferTokenUsage(50, 5, 0, PromptCache.Tier.FRESH, 500, 1_000);
        JinferTokenUsage sum = a.add(b);
        assertEquals(150, sum.inputTokenCount());
        assertEquals(15, sum.outputTokenCount());
        assertEquals(80, sum.cachedInputTokens());
        assertEquals(1_500, sum.promptNanos());
        assertEquals(3_000, sum.predictedNanos());
        assertNull(sum.servedFrom(), "an aggregate has no single serving");
        assertSame(a, a.add(null));
        // the superclass double-dispatches subclass-typed operands to add(); this exercises the
        // path that once recursed to StackOverflow inside an AiServices tool loop
        TokenUsage viaSuper = new TokenUsage(1, 1).add(a);
        assertEquals(101, viaSuper.inputTokenCount());
    }

    @Test
    void plainOperandsContributeCountsOnly() {
        JinferTokenUsage a =
                new JinferTokenUsage(100, 10, 80, PromptCache.Tier.SESSION, 1_000, 2_000);
        JinferTokenUsage sum = a.add(new TokenUsage(50, 5));
        assertEquals(150, sum.inputTokenCount());
        assertEquals(80, sum.cachedInputTokens(), "a plain usage has no cache read to add");
        assertEquals(1_000, sum.promptNanos(), "a plain usage has no timings to add");
    }

    @Test
    void toStringIsTheDiagnosisLine() {
        JinferTokenUsage usage =
                new JinferTokenUsage(
                        1204, 87, 1180, PromptCache.Tier.BLOCKS, 210_000_000L, 2_000_000_000L);
        String line = usage.toString();
        assertTrue(line.contains("input=1204"), line);
        assertTrue(line.contains("cached=1180, BLOCKS"), line);
        assertTrue(line.contains("prompt=0.21s"), line);
        assertTrue(line.contains("decode=43.5 tok/s"), line);
        // zero decode time (an empty completion) must not divide by zero
        String empty = new JinferTokenUsage(10, 0, 0, PromptCache.Tier.FRESH, 1_000, 0).toString();
        assertTrue(!empty.contains("tok/s"), empty);
    }
}

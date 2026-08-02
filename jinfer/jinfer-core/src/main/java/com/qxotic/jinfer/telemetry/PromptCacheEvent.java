package com.qxotic.jinfer.telemetry;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Period;
import jdk.jfr.StackTrace;

/**
 * Prompt cache health, sampled. {@link InferenceEvent#cachedTokens} says the cache did not help;
 * this says WHY, and the two causes want opposite fixes:
 *
 * <ul>
 *   <li>evictions near zero and hits low - the prompts genuinely diverge early, so fix the prompt
 *       (a timestamp near the top of a system prompt invalidates everything after it)
 *   <li>evictions climbing while hits fall - the cache is thrashing inside its budget, discarding
 *       blocks it was about to reuse, so raise the budget
 * </ul>
 *
 * <p>Counters are DELTAS since the previous sample, which keeps them small positive numbers (JFR
 * varint-encodes integers) and makes a rising trend visible directly rather than as the slope of an
 * ever-growing total.
 */
@Name("jinfer.PromptCache")
@Label("Prompt Cache")
@Category({"jinfer", "Memory"})
@Description(
        "Block cache health: rising evictions with falling hits means the budget is too small.")
@Period("1 s")
@StackTrace(false)
public final class PromptCacheEvent extends Event {

    @Label("Model")
    public String model;

    @Label("Blocks")
    public int blocks;

    @Label("Bytes")
    @DataAmount(DataAmount.BYTES)
    public long bytes;

    @Label("Budget")
    @DataAmount(DataAmount.BYTES)
    public long budgetBytes;

    @Label("Hits")
    public long hits;

    @Label("Misses")
    public long misses;

    @Label("Evictions")
    public long evictions;
}

package com.qxotic.jinfer.x.telemetry;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Period;
import jdk.jfr.StackTrace;

/**
 * Prompt cache health, sampled. {@link InferenceEvent#cachedTokens} says how much the cache helped;
 * this says why it did or did not, and the two causes want opposite fixes:
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
        "Prompt cache health: retained sessions, state allocation, block reuse and memory pressure.")
@Period("1 s")
@StackTrace(false)
public final class PromptCacheEvent extends Event {

    @Label("Model")
    public String model;

    @Label("Retained Sessions")
    public int retainedSessions;

    @Label("Retained Session Limit")
    public int retainedSessionLimit;

    @Label("Session Hits")
    public long sessionHits;

    @Label("State Allocations")
    public long stateAllocations;

    @Label("Session Snapshots")
    @DataAmount(DataAmount.BYTES)
    public long sessionSnapshotBytes;

    @Label("Blocks")
    public int blocks;

    @Label("Bytes")
    @DataAmount(DataAmount.BYTES)
    public long bytes;

    @Label("Budget")
    @DataAmount(DataAmount.BYTES)
    public long budgetBytes;

    @Label("Block Hits")
    public long blockHits;

    @Label("Block Misses")
    public long blockMisses;

    @Label("Block Evictions")
    public long blockEvictions;

    @Label("Block Discards")
    public long blockDiscards;

    @Label("Block Refusals")
    public long blockRefusals;
}

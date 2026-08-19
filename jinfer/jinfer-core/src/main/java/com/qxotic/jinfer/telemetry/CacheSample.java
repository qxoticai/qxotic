package com.qxotic.jinfer.telemetry;

/**
 * The prompt cache's health reading, as telemetry sees it. Telemetry owns this record - rather than
 * naming the cache's domain type - so events never force a module dependency: the engine maps its
 * {@code PromptCache.Sample} onto this vocabulary once, in the gauge lambda, and the event stream
 * stays acyclic. An exporter is a rename table, starting here.
 */
public record CacheSample(
        int retainedSessions,
        int retainedSessionLimit,
        long sessionHits,
        long stateAllocations,
        long sessionSnapshotBytes,
        int blocks,
        long bytes,
        long budgetBytes,
        long blockHits,
        long blockMisses,
        long blockEvictions,
        long blockDiscards,
        long blockRefusals) {}

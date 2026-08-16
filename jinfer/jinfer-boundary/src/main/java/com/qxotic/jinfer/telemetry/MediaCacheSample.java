package com.qxotic.jinfer.telemetry;

/**
 * The projected-media cache's health reading, as telemetry sees it - the same seam as {@link
 * CacheSample}: telemetry owns this vocabulary so the event stream never names the chat module's
 * {@code MediaEncodingCache}, and this package stays dependency-free.
 */
public record MediaCacheSample(
        int entries, long bytes, long budgetBytes, long hits, long misses, long refusals) {}

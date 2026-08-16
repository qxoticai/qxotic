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
 * Projected-media cache health, sampled. The media cache holds encoder-projected image/audio/video
 * batches keyed by content digest; the readings point at the fix:
 *
 * <ul>
 *   <li>refusals climbing - media bigger than the whole budget is served but never retained, so
 *       every use re-projects; raise {@code jinfer.mediaCacheMB} if the media is genuinely reused
 *   <li>misses high and hits low - the media is not recurring across requests at all, so the cache
 *       is not the problem; look at the encoder cost itself
 * </ul>
 *
 * <p>Counters are DELTAS since the previous sample, the same law as {@link PromptCacheEvent}.
 */
@Name("jinfer.MediaCache")
@Label("Media Cache")
@Category({"jinfer", "Memory"})
@Description("Projected-media cache health: retained entries, reuse and budget pressure.")
@Period("1 s")
@StackTrace(false)
public final class MediaCacheEvent extends Event {

    @Label("Model")
    public String model;

    @Label("Entries")
    public int entries;

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

    @Label("Refusals")
    public long refusals;
}

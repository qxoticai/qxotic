package com.qxotic.jinfer.x.telemetry;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * A model became resident: what it is, and what it costs. The event's duration is the load itself,
 * which separates a slow mmap from a slow first token.
 *
 * <p>Emitted by jinfer; consume it through JFR rather than constructing it. {@code dimensions} maps
 * to OpenTelemetry's {@code gen_ai.embeddings.dimension.count}, kept here rather than per call
 * because it is constant for the model's lifetime.
 */
@Name("jinfer.ModelLoad")
@Label("Model Load")
@Category({"jinfer", "Lifecycle"})
@Description("A model became resident: architecture, quantization and memory cost.")
@StackTrace(false)
public final class ModelLoadEvent extends Event {

    @Label("Model")
    public String model;

    @Label("Architecture")
    public String architecture;

    // Quantization belongs here, but general.file_type is an int enum with no name table in this
    // codebase and the honest source is the dominant tensor type. It lands with that lookup.

    @Label("Context Length")
    public int contextLength;

    /** Embedding models only; 0 elsewhere. */
    @Label("Dimensions")
    public int dimensions;

    @Label("Weights")
    @DataAmount(DataAmount.BYTES)
    public long weightsBytes;

    /** Memory-mapped rather than read: the first tokens page it in, so a cold run looks slow. */
    @Label("Memory Mapped")
    public boolean mapped;
}

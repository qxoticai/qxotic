package com.qxotic.jinfer.telemetry;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Period;
import jdk.jfr.StackTrace;

/**
 * How this JVM actually executes inference. Check it first when jinfer is slow: {@code vectorBits}
 * of 0 means the Java kernels fell back to scalar, either a missing {@code --add-modules
 * jdk.incubator.vector} or an explicit {@code -Djinfer.vectorBitSize=0}. That single fact explains
 * most "why is this slow" reports.
 *
 * <p>Sampled every chunk rather than once at startup, so a recording attached to an already-running
 * process is still self-describing.
 *
 * <p>The native gemm's resolved ISA and core plan belong here too - jam knows both and prints them
 * at startup - but it exposes neither to Java today. Those fields arrive with the accessor; adding
 * a field is a compatible change, and reporting the {@code JAM_ISA} override instead would be
 * reporting a request rather than the truth.
 */
@Name("jinfer.Runtime")
@Label("Runtime")
@Category({"jinfer", "Lifecycle"})
@Description("Vector API width and compute/decode thread counts actually in use.")
@Period("everyChunk")
@StackTrace(false)
public final class RuntimeEvent extends Event {

    /** Java Vector API width in bits; 0 means the scalar fallback. */
    @Label("Vector Bits")
    public int vectorBits;

    /** The one pool every kernel and jam backend runs on ({@code -Djinfer.threads}). */
    @Label("Threads")
    public int threads;
}

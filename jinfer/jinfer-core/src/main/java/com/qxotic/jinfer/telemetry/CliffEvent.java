package com.qxotic.jinfer.telemetry;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * One performance cliff report, emitted together with the log line - the log is for the human
 * watching the console, this event is for the recording, where it lands on the same timeline as GC
 * pauses, JIT compilation and thread state. Once per JVM run per cliff: the cliff's location is
 * statically known, so repeat events would only restate the first.
 */
@Name("jinfer.Cliff")
@Label("Performance Cliff")
@Category({"jinfer", "Performance"})
@Description("A performance cliff engaged (once per JVM run per cliff).")
@StackTrace(false)
public final class CliffEvent extends Event {

    /** The cliff's {@code PerformanceCliff} constant name. */
    @Label("Cliff")
    public String cliff;

    /** The cliff's full user-facing message, verbatim from the catalog. */
    @Label("Detail")
    public String detail;

    /** Emits the event. Cheap when no recording is active (JFR short-circuits). */
    public static void emit(String cliff, String detail) {
        CliffEvent event = new CliffEvent();
        event.cliff = cliff;
        event.detail = detail;
        event.commit();
    }
}

package com.qxotic.jinfer.telemetry;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * One decoded token. Off by default; enable it to see per-token latency on the same timeline as GC
 * pauses, JIT compilation and thread state - the question "what made my p99 spike" stops being a
 * hypothesis.
 *
 * <p>Deliberately carries NO fields. JFR's own {@code startTime}, {@code duration} and {@code
 * eventThread} are the entire payload, and this is the highest-frequency event jinfer emits, so it
 * gets the harshest budget.
 *
 * <p>A log probability would be the obvious field and belongs nowhere near here. It is
 * request-scoped data a caller consumes (to route on confidence, to score an eval), not telemetry,
 * and computing it costs a softmax per token - so switching this on to measure latency would change
 * the latency being measured. Telemetry must not perturb what it observes.
 */
@Name("jinfer.Decode")
@Label("Token Decode")
@Category({"jinfer", "Inference"})
@Description("One decoded token. Off by default; enable to correlate token latency with GC.")
@Enabled(false)
@StackTrace(false)
public final class DecodeEvent extends Event {}

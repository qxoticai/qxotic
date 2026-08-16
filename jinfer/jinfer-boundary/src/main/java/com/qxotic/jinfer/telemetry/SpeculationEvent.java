package com.qxotic.jinfer.telemetry;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * One speculative generation pass. Emitted only when speculative decoding actually runs, which is
 * why it is its own event rather than two permanently-zero counters on every {@link
 * InferenceEvent}.
 *
 * <p>It carries no model name: the emitting port knows its architecture, not which file was loaded,
 * and a mislabelled one is worse than none. It sits inside the {@link InferenceEvent} it belongs
 * to, on the same thread, so JFR's own timestamps and thread already correlate them.
 *
 * <p>{@code accepted / drafted} is the acceptance rate, and the only way to tell whether
 * speculation is paying for itself: every drafted token costs work whether or not it survives
 * verification, so a low rate means the draft model is being run for nothing. {@code forwards}
 * counts the verify passes, so {@code accepted / forwards} is tokens won per full-model forward.
 */
@Name("jinfer.Speculation")
@Label("Speculation")
@Category({"jinfer", "Inference"})
@Description("A speculative decoding pass: how many drafted tokens survived verification.")
@StackTrace(false)
public final class SpeculationEvent extends Event {

    @Label("Drafted Tokens")
    public int draftedTokens;

    @Label("Accepted Tokens")
    public int acceptedTokens;

    /** Verify passes through the full model. */
    @Label("Forwards")
    public int forwards;
}

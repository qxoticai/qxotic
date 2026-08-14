package com.qxotic.jinfer.x.boundary;

/**
 * How an utterance is spoken, as opposed to {@link Config}, which is what the model IS.
 *
 * <p>Deliberately NARROW: exactly what a framework request object can express, so an integration
 * passes it through without knowing which port is underneath. Everything else a speech model takes
 * - a latent noise scale, a language, a lexicon - is chosen ONCE and belongs on that port's own
 * loader, where it can be typed and range-checked. A port REJECTS a knob it cannot honour rather
 * than substituting its default: a caller who passed one and got the default has been lied to.
 *
 * <p>An interface, not a record, so a port may extend it with knobs of its own for callers that do
 * name the port.
 */
public interface SpeechOptions {

    /**
     * Rate multiplier, 1.0 = the model's natural rate; null = the model's default. The one knob
     * both framework request objects name, so it is the one an adapter can pass through blind.
     * Ports bound it - a rate that multiplies predicted durations is also a cost multiplier.
     */
    Double speed();

    /** Every knob at the model's own default. */
    SpeechOptions NONE = () -> null;

    static SpeechOptions speed(double speed) {
        // the one knob both frameworks pass through blind, so the funnel is where it is bounded:
        // a non-positive or non-finite rate multiplies predicted durations into garbage
        if (!Double.isFinite(speed) || speed <= 0)
            throw new IllegalArgumentException(
                    "speed must be a positive finite number, got " + speed);
        return () -> speed;
    }
}

package com.qxotic.jinfer.x.server;

import java.util.function.Consumer;

/**
 * The live output channels of a streaming generation: the text and reasoning sinks. They travel
 * together so {@link Generation} takes one parameter instead of loose nulls for the common
 * non-streaming case ({@link #NONE}). Either channel may be null (completions have no reasoning
 * lane).
 *
 * <p>There is deliberately no running token counter here. One existed, and the pipeline only filled
 * it AFTER the pass completed - so every delta chunk carried a {@code usage} object reading all
 * zeros, a non-standard field that was also never true. Real usage rides the terminal chunk, built
 * from the finished {@link Reply}.
 */
record Sinks(Consumer<String> onText, Consumer<String> onReasoning) {

    /** No streaming: the result carries the full text and usage, so both channels are absent. */
    static final Sinks NONE = new Sinks(null, null);

    /** A single text channel (completions and the Responses API). */
    static Sinks text(Consumer<String> onText) {
        return new Sinks(onText, null);
    }
}

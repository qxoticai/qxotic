package com.qxotic.jinfer;

import java.util.function.Consumer;

/**
 * The live output channels of a streaming generation: the text / reasoning / tool-call sinks and the
 * running usage counter. They always travel together — bundling them keeps {@link Generation}'s API
 * at three parameters and stops the transport layer from passing four loose nulls for the common
 * non-streaming case ({@link #NONE}). Any channel may be null (e.g. completions have no reasoning or
 * tool-call sink); {@code usage} is null when running usage is not tracked.
 */
record Sinks(Consumer<String> onText, Consumer<String> onReasoning, Consumer<String> onToolCall,
             OpenAiSchema.Usage usage) {

    /** No streaming: the result carries the full text/usage, so every channel is absent. */
    static final Sinks NONE = new Sinks(null, null, null, null);

    /** A single text channel with usage tracking (completions and the Responses API). */
    static Sinks text(Consumer<String> onText, OpenAiSchema.Usage usage) {
        return new Sinks(onText, null, null, usage);
    }
}

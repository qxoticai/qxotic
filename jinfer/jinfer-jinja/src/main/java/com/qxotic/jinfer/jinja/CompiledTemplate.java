package com.qxotic.jinfer.jinja;

import java.util.Map;

/**
 * A compiled Jinja chat template: vars in, rendered String out - produced by {@link
 * JinjaRenderer#template(String)}. This is the raw render seam; lowering a conversation to
 * ingest-ready batches is the chat layer's ChatTemplate contract, not this one.
 */
@FunctionalInterface
public interface CompiledTemplate {
    /**
     * Renders against {@code vars} - the chat-template context (typically {@code messages}, {@code
     * add_generation_prompt}, {@code tools}). Nested values may be {@link Map}s, {@link
     * java.util.List}s, strings, numbers and booleans.
     */
    String render(Map<String, Object> vars);
}

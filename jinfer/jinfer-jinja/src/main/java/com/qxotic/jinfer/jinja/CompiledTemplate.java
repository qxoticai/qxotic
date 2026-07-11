package com.qxotic.jinfer.jinja;

import java.util.Map;

/**
 * A compiled Jinja chat template: vars in, rendered String out - produced by {@link
 * JinjaRenderer#template(String)}. This is the raw render seam; lowering a conversation to
 * ingest-ready batches is the chat layer's ChatTemplate contract, not this one.
 */
@FunctionalInterface
public interface CompiledTemplate {
    String render(Map<String, Object> vars);
}

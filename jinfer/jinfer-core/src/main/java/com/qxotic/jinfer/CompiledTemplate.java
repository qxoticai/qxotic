package com.qxotic.jinfer;

import java.util.Map;

/**
 * A compiled Jinja chat template: vars in, rendered String out — produced by {@code
 * JinjaRenderer::template}. This is the raw render seam, distinct from {@link
 * com.qxotic.jinfer.chat.ChatTemplate}, the batch-producing chat contract.
 */
@FunctionalInterface
public interface CompiledTemplate {
    String render(Map<String, Object> vars);
}

package com.qxotic.jinfer;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/** A compiled Jinja chat template: an opaque parsed program plus its renderer, produced by
 *  {@code JinjaRenderer::template}. This is the raw render seam (vars in, String out) - distinct
 *  from {@link com.qxotic.jinfer.chat.ChatTemplate}, the batch-producing chat contract. */
public final class CompiledTemplate {
    final Object compiled;
    private final BiFunction<Object, Map<String, Object>, String> renderer;

    public CompiledTemplate(Object compiled, BiFunction<Object, Map<String, Object>, String> renderer) {
        this.compiled = Objects.requireNonNull(compiled);
        this.renderer = Objects.requireNonNull(renderer);
    }

    public String render(Map<String, Object> vars) {
        return renderer.apply(compiled, vars);
    }
}

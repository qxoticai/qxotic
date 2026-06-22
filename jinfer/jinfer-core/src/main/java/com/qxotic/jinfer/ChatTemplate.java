package com.qxotic.jinfer;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

public final class ChatTemplate {
    final Object compiled;
    private final BiFunction<Object, Map<String, Object>, String> renderer;

    public ChatTemplate(Object compiled, BiFunction<Object, Map<String, Object>, String> renderer) {
        this.compiled = Objects.requireNonNull(compiled);
        this.renderer = Objects.requireNonNull(renderer);
    }

    public String render(Map<String, Object> vars) {
        return renderer.apply(compiled, vars);
    }
}

package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.BiConsumer;
import org.springframework.util.StringUtils;

/** The {@code <prefix>.companions.*} map onto a builder: each value a model ref or a local path. */
final class Companions {
    private Companions() {}

    static void attach(
            String prefix,
            Map<String, String> companions,
            BiConsumer<String, String> ref,
            BiConsumer<String, Path> path) {
        if (companions == null) return;
        companions.forEach(
                (capability, value) -> {
                    String key = prefix + ".companions." + capability;
                    if (!StringUtils.hasText(value)) {
                        throw new IllegalStateException(key + " must not be blank");
                    }
                    if (value.contains("://")) {
                        throw new IllegalStateException(
                                key + " is a URL; download it first and configure its local path");
                    }
                    if (ModelStore.isRef(value)) ref.accept(capability, value);
                    else path.accept(capability, Path.of(value));
                });
    }
}

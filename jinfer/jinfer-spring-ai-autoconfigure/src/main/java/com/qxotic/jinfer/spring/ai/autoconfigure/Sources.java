package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import org.springframework.util.StringUtils;

/** Where a property names a model file: a model ref, resolved by the builder, or a local path. */
final class Sources {
    private Sources() {}

    /**
     * {@code value} of property {@code key} onto a builder: a ref to {@code ref}, else a path to
     * {@code path}. A URL is refused: the starter never downloads from one.
     */
    static void model(String key, String value, Consumer<String> ref, Consumer<Path> path) {
        if (value.contains("://")) {
            try {
                ModelStore.requireRef(value);
            } catch (IllegalArgumentException url) { // naming the ref a hub link spells
                throw new IllegalStateException(key + ": " + url.getMessage(), url);
            }
        }
        if (ModelStore.isRef(value)) ref.accept(value);
        else path.accept(Path.of(value));
    }

    /** The {@code <prefix>.companions.*} map onto a builder, each value as {@link #model}. */
    static void companions(
            String prefix,
            Map<String, String> companions,
            BiConsumer<String, String> ref,
            BiConsumer<String, Path> path) {
        if (companions == null) return;
        companions.forEach(
                (capability, value) -> {
                    String key = prefix + ".companions." + capability;
                    if (!StringUtils.hasText(value))
                        throw new IllegalStateException(key + " must not be blank");
                    model(
                            key,
                            value,
                            v -> ref.accept(capability, v),
                            p -> path.accept(capability, p));
                });
    }
}

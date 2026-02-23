package com.qxotic.tokenizers.gguf;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public final class GGUFPreTokenizerRegistry {

    private final Map<String, Splitter> splitters;

    private GGUFPreTokenizerRegistry(Map<String, Splitter> splitters) {
        this.splitters = splitters;
    }

    public static GGUFPreTokenizerRegistry defaults() {
        GGUFPreTokenizerRegistry registry = new GGUFPreTokenizerRegistry(new ConcurrentHashMap<>());
        GGUFSplitters.registerDefaults(registry);
        return registry;
    }

    public GGUFPreTokenizerRegistry register(String name, Splitter splitter) {
        splitters.put(normalizeName(name), Objects.requireNonNull(splitter, "splitter"));
        return this;
    }

    public boolean contains(String name) {
        return splitters.containsKey(normalizeName(name));
    }

    public Splitter resolve(String name) {
        if (name == null || name.isEmpty()) {
            return null;
        }
        return splitters.get(normalizeName(name));
    }

    public Splitter require(String name) {
        Splitter splitter = resolve(name);
        if (splitter == null) {
            throw new GGUFTokenizerException(
                    "Unknown GGUF pre-tokenizer: "
                            + name
                            + ". Registered values: "
                            + names().stream().sorted().reduce((a, b) -> a + ", " + b).orElse(""));
        }
        return splitter;
    }

    public Set<String> names() {
        return Collections.unmodifiableSet(splitters.keySet());
    }

    public Map<String, Splitter> snapshot() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(splitters));
    }

    private static String normalizeName(String name) {
        Objects.requireNonNull(name, "name");
        return name.trim().toLowerCase(Locale.ROOT);
    }
}

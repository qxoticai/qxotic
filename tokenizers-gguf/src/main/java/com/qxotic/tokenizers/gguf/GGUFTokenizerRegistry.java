package com.qxotic.tokenizers.gguf;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public final class GGUFTokenizerRegistry {

    private final Map<String, GGUFTokenizerFactory> tokenizers;

    private GGUFTokenizerRegistry(Map<String, GGUFTokenizerFactory> tokenizers) {
        this.tokenizers = tokenizers;
    }

    public static GGUFTokenizerRegistry defaults() {
        GGUFTokenizerRegistry registry = new GGUFTokenizerRegistry(new ConcurrentHashMap<>());
        GGUFTokenizerFactories.registerDefaults(registry);
        return registry;
    }

    public GGUFTokenizerRegistry register(String name, GGUFTokenizerFactory factory) {
        tokenizers.put(normalizeName(name), Objects.requireNonNull(factory, "factory"));
        return this;
    }

    public boolean contains(String name) {
        return tokenizers.containsKey(normalizeName(name));
    }

    public GGUFTokenizerFactory resolve(String name) {
        if (name == null || name.isEmpty()) {
            return null;
        }
        return tokenizers.get(normalizeName(name));
    }

    public GGUFTokenizerFactory require(String name) {
        GGUFTokenizerFactory factory = resolve(name);
        if (factory == null) {
            throw new GGUFTokenizerException(
                    "Unknown GGUF tokenizer model: "
                            + name
                            + ". Registered values: "
                            + names().stream().sorted().reduce((a, b) -> a + ", " + b).orElse(""));
        }
        return factory;
    }

    public Set<String> names() {
        return Collections.unmodifiableSet(tokenizers.keySet());
    }

    public Map<String, GGUFTokenizerFactory> snapshot() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(tokenizers));
    }

    private static String normalizeName(String name) {
        Objects.requireNonNull(name, "name");
        return name.trim().toLowerCase(Locale.ROOT);
    }
}

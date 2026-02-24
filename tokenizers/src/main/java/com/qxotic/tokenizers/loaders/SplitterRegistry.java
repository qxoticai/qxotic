package com.qxotic.tokenizers.loaders;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for managing model-specific text splitters.
 *
 * <p>This registry allows registration and lookup of splitters by name, with support for the
 * standard model families (llama, qwen, mistral, etc.).
 */
public final class SplitterRegistry {

    private final Map<String, Splitter> splitters;

    private SplitterRegistry(Map<String, Splitter> splitters) {
        this.splitters = splitters;
    }

    /**
     * Creates a new registry with default model splitters pre-registered.
     *
     * @return a new registry with defaults
     */
    public static SplitterRegistry defaults() {
        SplitterRegistry registry = new SplitterRegistry(new ConcurrentHashMap<>());
        registerDefaults(registry);
        return registry;
    }

    /**
     * Creates a new empty registry.
     *
     * @return a new empty registry
     */
    public static SplitterRegistry empty() {
        return new SplitterRegistry(new ConcurrentHashMap<>());
    }

    private static void registerDefaults(SplitterRegistry registry) {
        registry.register("llama", ModelSplitters.LLAMA3);
        registry.register("llama-bpe", ModelSplitters.LLAMA3);
        registry.register("mistral", ModelSplitters.LLAMA3);
        registry.register("mixtral", ModelSplitters.LLAMA3);
        registry.register("phi", ModelSplitters.LLAMA3);
        registry.register("phi3", ModelSplitters.LLAMA3);
        registry.register("dbrx", ModelSplitters.LLAMA3);

        registry.register("qwen", ModelSplitters.QWEN2);
        registry.register("qwen2", ModelSplitters.QWEN2);
        registry.register("qwen3", ModelSplitters.QWEN2);

        registry.register("smollm", ModelSplitters.SMOLLM2);
        registry.register("smollm2", ModelSplitters.SMOLLM2);

        registry.register("tekken", ModelSplitters.TEKKEN);
        registry.register("refact", ModelSplitters.REFACT);
        registry.register("granite", ModelSplitters.REFACT);
        registry.register("default", ModelSplitters.DEFAULT_BPE);

        registry.register("gemma", ModelSplitters.IDENTITY);
        registry.register("gemma2", ModelSplitters.IDENTITY);
        registry.register("gemma3", ModelSplitters.IDENTITY);
    }

    /**
     * Registers a splitter with the given name.
     *
     * @param name the name to register under
     * @param splitter the splitter to register
     * @return this registry for method chaining
     * @throws NullPointerException if name or splitter is null
     */
    public SplitterRegistry register(String name, Splitter splitter) {
        splitters.put(normalizeName(name), Objects.requireNonNull(splitter, "splitter"));
        return this;
    }

    /**
     * Checks if a splitter is registered for the given name.
     *
     * @param name the name to check
     * @return true if registered, false otherwise
     */
    public boolean contains(String name) {
        return splitters.containsKey(normalizeName(name));
    }

    /**
     * Looks up a splitter by name.
     *
     * @param name the name to look up
     * @return the splitter, or null if not found
     */
    public Splitter resolve(String name) {
        if (name == null || name.isEmpty()) {
            return null;
        }
        return splitters.get(normalizeName(name));
    }

    /**
     * Looks up a splitter by name, throwing an exception if not found.
     *
     * @param name the name to look up
     * @return the splitter
     * @throws IllegalArgumentException if not found
     */
    public Splitter require(String name) {
        Splitter splitter = resolve(name);
        if (splitter == null) {
            throw new IllegalArgumentException(
                    "Unknown splitter: "
                            + name
                            + ". Registered: "
                            + names().stream()
                                    .sorted()
                                    .reduce((a, b) -> a + ", " + b)
                                    .orElse("(none)"));
        }
        return splitter;
    }

    /**
     * Returns all registered names.
     *
     * @return set of registered names
     */
    public Set<String> names() {
        return Collections.unmodifiableSet(splitters.keySet());
    }

    /**
     * Returns a snapshot of all registered splitters.
     *
     * @return map of names to splitters
     */
    public Map<String, Splitter> snapshot() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(splitters));
    }

    private static String normalizeName(String name) {
        Objects.requireNonNull(name, "name");
        return name.trim().toLowerCase(Locale.ROOT);
    }
}

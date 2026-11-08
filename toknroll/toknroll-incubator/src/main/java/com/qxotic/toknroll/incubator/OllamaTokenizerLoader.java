package com.qxotic.toknroll.incubator;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Loads the tokenizer of a model Ollama has pulled, from its local store, by the name {@code ollama
 * run} takes: {@code [host/][namespace/]model[:tag]}, host {@code registry.ollama.ai}, namespace
 * {@code library} and tag {@code latest} when left out, matched without regard to case as Ollama
 * does ({@code qwen3:4b}, {@code hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M}). The store is
 * {@code $OLLAMA_MODELS}, else {@code ~/.ollama/models}; nothing is downloaded.
 */
public final class OllamaTokenizerLoader {
    private static final String MODEL_LAYER = "application/vnd.ollama.image.model";

    // Ollama's own name rules: host, namespace and model as in types/model/name.go, then the tag.
    private static final Pattern NAME =
            Pattern.compile(
                    "(?:([A-Za-z0-9][A-Za-z0-9_.:-]{0,349})/)??"
                            + "(?:([A-Za-z0-9][A-Za-z0-9_-]{0,79})/)?"
                            + "([A-Za-z0-9][A-Za-z0-9_.-]{0,79})"
                            + "(?::([A-Za-z0-9_][A-Za-z0-9_.-]{0,79}))?");
    private static final Pattern DIGEST = Pattern.compile("sha256:([0-9a-f]{64})");

    private final GGUFTokenizerLoader ggufLoader;

    public OllamaTokenizerLoader() {
        this(GGUFTokenizerLoader.createBuilderWithBuiltins().build());
    }

    public OllamaTokenizerLoader(GGUFTokenizerLoader ggufLoader) {
        this.ggufLoader = Objects.requireNonNull(ggufLoader, "ggufLoader");
    }

    /** The tokenizer of {@code name} in the default store. */
    public Tokenizer fromLocal(String name) {
        String configured = System.getenv("OLLAMA_MODELS");
        return fromLocal(
                configured != null && !configured.isBlank()
                        ? Path.of(configured)
                        : Path.of(System.getProperty("user.home"), ".ollama", "models"),
                name);
    }

    /** The tokenizer of {@code name} in the Ollama store at {@code store} (its models folder). */
    public Tokenizer fromLocal(Path store, String name) {
        return ggufLoader.fromLocal(modelBlob(store, name));
    }

    /** The GGUF blob holding the model {@code name} names in {@code store}. */
    static Path modelBlob(Path store, String name) {
        Matcher m = NAME.matcher(Objects.requireNonNull(name, "name"));
        if (!m.matches()) {
            throw new IllegalArgumentException(
                    "not an Ollama model name ([host/][namespace/]model[:tag]): '" + name + "'");
        }
        String host = m.group(1) != null ? m.group(1) : "registry.ollama.ai";
        String namespace = m.group(2) != null ? m.group(2) : "library";
        String tag = m.group(4) != null ? m.group(4) : "latest";
        Path root = store.toAbsolutePath().normalize();

        Path model = child(child(child(root.resolve("manifests"), host), namespace), m.group(3));
        Path manifest = child(model, tag);
        if (manifest == null || !Files.isRegularFile(manifest)) {
            String full = host + "/" + namespace + "/" + m.group(3) + ":" + tag;
            List<String> tags = entries(model);
            throw new IllegalArgumentException(
                    tags.isEmpty()
                            ? "Ollama has not pulled " + full + " (store " + root + ")"
                            : "Ollama has no "
                                    + full
                                    + "; pulled tags: "
                                    + String.join(", ", tags));
        }

        // the model layer, the GGUF Ollama runs; a projector or adapter layer is GGUF too
        String digest = null;
        try {
            for (Object layer : (List<?>) Json.parseMap(Files.readString(manifest)).get("layers")) {
                Map<?, ?> l = (Map<?, ?>) layer;
                if (MODEL_LAYER.equals(l.get("mediaType"))) {
                    digest = (String) l.get("digest");
                    break;
                }
            }
        } catch (IOException | RuntimeException e) {
            throw new TokenizerLoadException("unreadable Ollama manifest " + manifest, e);
        }
        Matcher d = DIGEST.matcher(String.valueOf(digest));
        if (!d.matches()) {
            throw new TokenizerLoadException(
                    "Ollama manifest " + manifest + " has no model layer with a sha256 digest");
        }
        Path blob = root.resolve("blobs").resolve("sha256-" + d.group(1));
        if (!Files.isRegularFile(blob)) {
            throw new TokenizerLoadException("Ollama blob " + blob + " is missing");
        }
        return blob;
    }

    /** {@code dir}'s entry named {@code name}, ignoring case as Ollama does; null when none. */
    private static Path child(Path dir, String name) {
        for (String entry : entries(dir)) {
            if (entry.equalsIgnoreCase(name)) {
                return dir.resolve(entry);
            }
        }
        return null;
    }

    private static List<String> entries(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> s = Files.list(dir)) {
            return s.map(p -> p.getFileName().toString()).sorted().collect(Collectors.toList());
        } catch (IOException e) {
            return List.of();
        }
    }
}

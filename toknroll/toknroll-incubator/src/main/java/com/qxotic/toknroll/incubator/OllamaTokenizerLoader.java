package com.qxotic.toknroll.incubator;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Loads GGUF tokenizers from local Ollama model manifests and blobs. */
public final class OllamaTokenizerLoader {
    private static final Pattern OLLAMA_DIGEST_PATTERN =
            Pattern.compile("\"digest\"\\s*:\\s*\"sha256:([0-9a-fA-F]{64})\"");

    private final GGUFTokenizerLoader ggufLoader;

    public OllamaTokenizerLoader() {
        this(GGUFTokenizerLoader.createBuilderWithBuiltins().build());
    }

    public OllamaTokenizerLoader(GGUFTokenizerLoader ggufLoader) {
        this.ggufLoader = Objects.requireNonNull(ggufLoader, "ggufLoader");
    }

    public Tokenizer fromLocal(String modelTag) {
        return fromLocal(resolveOllamaModelsRoot(), modelTag);
    }

    public Tokenizer fromLocal(Path ollamaModelsRoot, String modelTag) {
        Objects.requireNonNull(ollamaModelsRoot, "ollamaModelsRoot");
        String normalizedTag = normalizeOllamaTag(modelTag);
        Path modelsRoot = ollamaModelsRoot.toAbsolutePath().normalize();
        Path manifest = resolveOllamaManifest(modelsRoot, normalizedTag);
        Path ggufFile = resolveGgufBlobFromManifest(modelsRoot, manifest);
        return ggufLoader.fromLocal(ggufFile);
    }

    private static Path resolveOllamaModelsRoot() {
        String configured = System.getenv("OLLAMA_MODELS");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured);
        }
        String home = System.getProperty("user.home", ".");
        return Path.of(home, ".ollama", "models");
    }

    private static String normalizeOllamaTag(String modelTag) {
        if (modelTag == null || modelTag.isBlank()) {
            throw new IllegalArgumentException("modelTag must not be blank");
        }
        String normalized = modelTag.trim();
        if (normalized.contains("\\") || normalized.contains("..")) {
            throw new IllegalArgumentException(
                    "modelTag contains invalid path characters: " + modelTag);
        }
        return normalized;
    }

    private static Path resolveOllamaManifest(Path modelsRoot, String modelTag) {
        String model = modelTag;
        String tag = "latest";
        int colon = modelTag.lastIndexOf(':');
        if (colon > 0 && colon < modelTag.length() - 1) {
            model = modelTag.substring(0, colon);
            tag = modelTag.substring(colon + 1);
        }

        Path manifestsRoot = modelsRoot.resolve("manifests");
        Path[] candidates =
                new Path[] {
                    manifestsRoot.resolve("registry.ollama.ai").resolve("library").resolve(model).resolve(tag),
                    manifestsRoot.resolve("registry.ollama.ai").resolve(model).resolve(tag),
                    manifestsRoot.resolve(model).resolve(tag)
                };
        for (Path candidate : candidates) {
            if (Files.exists(candidate) && Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        throw new IllegalArgumentException(
                "Could not locate Ollama manifest for model tag '"
                        + modelTag
                        + "' under "
                        + manifestsRoot);
    }

    private static Path resolveGgufBlobFromManifest(Path modelsRoot, Path manifestPath) {
        String manifest;
        try {
            manifest = Files.readString(manifestPath);
        } catch (IOException e) {
            throw new TokenizerLoadException("Failed to read Ollama manifest " + manifestPath, e);
        }

        Matcher matcher = OLLAMA_DIGEST_PATTERN.matcher(manifest);
        Path blobsRoot = modelsRoot.resolve("blobs");
        while (matcher.find()) {
            String digest = matcher.group(1).toLowerCase(Locale.ROOT);
            Path blobPath = blobsRoot.resolve("sha256-" + digest);
            if (!Files.exists(blobPath) || !Files.isRegularFile(blobPath)) {
                continue;
            }
            if (isGgufFile(blobPath)) {
                return blobPath;
            }
        }

        throw new IllegalArgumentException(
                "No GGUF blob found in Ollama manifest " + manifestPath + " under " + blobsRoot);
    }

    private static boolean isGgufFile(Path blobPath) {
        try (InputStream in = Files.newInputStream(blobPath)) {
            byte[] magic = new byte[4];
            int read = in.read(magic);
            return read == 4 && magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
        } catch (IOException e) {
            return false;
        }
    }
}

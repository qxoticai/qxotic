package com.qxotic.toknroll.incubator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.TokenizerLoadException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Name resolution against a fake Ollama store laid out as Ollama writes it. */
class OllamaTokenizerLoaderTest {
    @TempDir Path store;

    private String model;
    private String projector;

    @BeforeEach
    void pull() throws IOException {
        model = blob("GGUF model");
        projector = blob("GGUF projector");
        // the projector is GGUF too, and listed first: only the model layer may be picked
        String manifest =
                "{\"schemaVersion\":2,\"layers\":["
                        + layer("application/vnd.ollama.image.projector", projector)
                        + ","
                        + layer("application/vnd.ollama.image.model", model)
                        + ","
                        + layer("application/vnd.ollama.image.template", blob("{{ .Prompt }}"))
                        + "]}";
        manifest("registry.ollama.ai/library/qwen3/4b", manifest);
        manifest("registry.ollama.ai/library/qwen3/latest", manifest);
        manifest("registry.ollama.ai/alice/Mixed-Case/v1", manifest);
        manifest("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF/Q4_K_M", manifest);
    }

    @Test
    void namesResolveAsOllamaResolvesThem() {
        for (String name :
                new String[] {
                    "qwen3:4b",
                    "qwen3",
                    "library/qwen3:4b",
                    "registry.ollama.ai/library/qwen3:4b",
                    "QWEN3:4B",
                    "alice/mixed-case:V1",
                    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
                }) {
            assertEquals(
                    "sha256-" + model,
                    OllamaTokenizerLoader.modelBlob(store, name).getFileName().toString(),
                    name);
        }
    }

    @Test
    void aMissingTagNamesThePulledOnes() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> OllamaTokenizerLoader.modelBlob(store, "qwen3:8b"));
        assertTrue(e.getMessage().contains("pulled tags: 4b, latest"), e.getMessage());
        e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> OllamaTokenizerLoader.modelBlob(store, "llama3.2"));
        assertTrue(e.getMessage().contains("has not pulled"), e.getMessage());
    }

    @Test
    void aPathIsNotAName() {
        for (String name :
                new String[] {
                    "/etc/passwd", "../qwen3:4b", "qwen3/../../x", "", "a:b:c", " qwen3:4b"
                }) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> OllamaTokenizerLoader.modelBlob(store, name),
                    name);
        }
    }

    @Test
    void aMissingBlobOrModelLayerIsRefused() throws IOException {
        manifest(
                "registry.ollama.ai/library/gone/latest",
                "{\"layers\":["
                        + layer("application/vnd.ollama.image.model", "0".repeat(64))
                        + "]}");
        manifest(
                "registry.ollama.ai/library/textless/latest",
                "{\"layers\":[" + layer("application/vnd.ollama.image.template", model) + "]}");
        manifest("registry.ollama.ai/library/garbled/latest", "{\"layers\": 7}");
        for (String name : new String[] {"gone", "textless", "garbled"}) {
            assertThrows(
                    TokenizerLoadException.class,
                    () -> OllamaTokenizerLoader.modelBlob(store, name),
                    name);
        }
    }

    private String blob(String content) throws IOException {
        String hex;
        try {
            hex =
                    HexFormat.of()
                            .formatHex(
                                    MessageDigest.getInstance("SHA-256")
                                            .digest(content.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        Path p = store.resolve("blobs").resolve("sha256-" + hex);
        Files.createDirectories(p.getParent());
        Files.writeString(p, content);
        return hex;
    }

    private void manifest(String path, String json) throws IOException {
        Path p = store.resolve("manifests").resolve(path);
        Files.createDirectories(p.getParent());
        Files.writeString(p, json);
    }

    private static String layer(String mediaType, String hex) {
        return "{\"mediaType\":\"" + mediaType + "\",\"digest\":\"sha256:" + hex + "\",\"size\":1}";
    }
}

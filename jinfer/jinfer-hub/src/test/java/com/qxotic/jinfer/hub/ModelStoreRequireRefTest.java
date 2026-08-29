package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** {@link ModelStore#requireRef}: the one door that decides what a model ref is, and teaches it. */
class ModelStoreRequireRefTest {

    // ---- accepted: a model ref names its host ----

    @Test
    void acceptsHuggingFaceRef() {
        assertDoesNotThrow(() -> ModelStore.requireRef("hf.co/owner/repo:Q8_0"));
    }

    @Test
    void acceptsModelScopeRef() {
        assertDoesNotThrow(() -> ModelStore.requireRef("modelscope.cn/owner/repo:Q8_0"));
    }

    @Test
    void acceptsARefWithoutAQuant() {
        assertDoesNotThrow(() -> ModelStore.requireRef("hf.co/owner/repo"));
    }

    @Test
    void acceptsARefWithAPathAndRevision() {
        assertDoesNotThrow(() -> ModelStore.requireRef("hf.co/owner/repo@main/sub/model.gguf"));
    }

    @Test
    void rejectsHuggingFaceBrowserSpelling() {
        String message =
                message(() -> ModelStore.requireRef("https://huggingface.co/owner/repo:Q8_0"));
        assertTrue(message.contains("URL"));
        assertTrue(message.contains("modelPath"));
    }

    // ---- rejected, each with its own remedy ----

    @Test
    void rejectsNull() {
        assertTrue(message(() -> ModelStore.requireRef(null)).contains("a model ref is required"));
    }

    @Test
    void rejectsBlank() {
        assertTrue(message(() -> ModelStore.requireRef("   ")).contains("a model ref is required"));
    }

    @Test
    void rejectsAUrlWithTheModelPathRemedy() {
        String message = message(() -> ModelStore.requireRef("https://example.org/model.gguf"));
        assertTrue(message.contains("URL"));
        assertTrue(message.contains("modelPath"));
    }

    @Test
    void rejectsAnAbsoluteLocalPath() {
        String message = message(() -> ModelStore.requireRef("/models/x.gguf"));
        assertTrue(message.contains("local path"));
        assertTrue(message.contains("modelPath"));
    }

    @Test
    void rejectsARelativeLocalPath() {
        String message = message(() -> ModelStore.requireRef("./models/x.gguf"));
        assertTrue(message.contains("local path"));
        assertTrue(message.contains("modelPath"));
    }

    @Test
    void rejectsAHomePath() {
        String message = message(() -> ModelStore.requireRef("~/models/x.gguf"));
        assertTrue(message.contains("local path"));
        assertTrue(message.contains("modelPath"));
    }

    @Test
    void rejectsAWindowsPath() {
        String message = message(() -> ModelStore.requireRef("C:\\models\\x.gguf"));
        assertTrue(message.contains("local path"));
        assertTrue(message.contains("modelPath"));
    }

    @Test
    void everyPathShapedStringIsSentToModelPath() {
        for (String path :
                new String[] {
                    "/models/x.gguf", "./models/x.gguf", "~/models/x.gguf",
                    "C:\\models\\x.gguf", "model.gguf", "models/llama.gguf"
                }) {
            assertTrue(message(() -> ModelStore.requireRef(path)).contains("modelPath"), path);
        }
    }

    @Test
    void acceptsBareOwnerRepo() {
        assertDoesNotThrow(() -> ModelStore.requireRef("owner/repo:Q8_0"));
        assertDoesNotThrow(() -> ModelStore.requireRef("owner/repo"));
        assertDoesNotThrow(() -> ModelStore.requireRef("owner/repo@main/sub/model.gguf"));
    }

    @Test
    void rejectsAnUnknownHostWithTheKnownOnes() {
        String message = message(() -> ModelStore.requireRef("example.org/owner/repo:Q8_0"));
        assertTrue(message.contains("does not name a known host"));
        assertTrue(message.contains("hf.co/"));
        assertTrue(message.contains("modelscope.cn/"));
    }

    @Test
    void rejectsAStringWithTooFewSegments() {
        assertTrue(
                message(() -> ModelStore.requireRef("justonething")).contains("not a model ref"));
    }

    private static String message(Runnable action) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, action::run);
        return e.getMessage();
    }
}

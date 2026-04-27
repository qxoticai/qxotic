package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class GGUFTokenizerLoaderApiTest {

    @Test
    void builderMethodsCreateLoaders() {
        assertDoesNotThrow(() -> GGUFTokenizerLoader.builderDefault().build());
        assertDoesNotThrow(() -> GGUFTokenizerLoader.builderEmpty().build());
    }

    @Test
    void fromLocalRejectsNonGgufFile() throws Exception {
        Path temp = Files.createTempFile("toknroll-gguf", ".txt");
        try {
            GGUFTokenizerLoader loader = GGUFTokenizerLoader.builderDefault().build();
            assertThrows(IllegalArgumentException.class, () -> loader.fromLocal(temp));
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @Test
    void fromHuggingFaceRequiresExactGgufPath() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.builderDefault().build();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        loader.fromHuggingFace(
                                "Qwen", "Qwen3-0.6B-GGUF", "main", "weights.bin", true, false));
    }
}

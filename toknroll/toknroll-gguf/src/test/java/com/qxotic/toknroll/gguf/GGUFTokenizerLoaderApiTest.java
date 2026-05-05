package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GGUFTokenizerLoaderApiTest {

    @Test
    void builderMethodsCreateLoaders() {
        assertNotNull(GGUFTokenizerLoader.createBuilderWithBuiltins().build());
        assertNotNull(GGUFTokenizerLoader.createEmptyBuilder().build());
    }

    @Test
    void fromLocalRejectsNonGgufFile(@TempDir Path tempDir) throws Exception {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Path file = tempDir.resolve("not-gguf.txt");
        Files.writeString(file, "content");
        assertThrows(IllegalArgumentException.class, () -> loader.fromLocal(file));
    }

    @Test
    void fromHuggingFaceRequiresExactGgufPath() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        assertThrows(
                RuntimeException.class,
                () -> loader.fromHuggingFace("unsloth", "nonexistent", "not-a-gguf.txt"));
    }
}

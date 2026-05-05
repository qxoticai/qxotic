package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.toknroll.TokenizationModel;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GGUFTokenizerLoaderEdgeCaseTest {

    @Test
    void fromLocalRejectsNonexistentFile() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        assertThrows(
                IllegalArgumentException.class,
                () -> loader.fromLocal(Path.of("/nonexistent.gguf")));
    }

    @Test
    void fromLocalRejectsDirectory(@TempDir Path tempDir) throws Exception {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Path dir = tempDir.resolve("mydir");
        Files.createDirectories(dir);
        assertThrows(IllegalArgumentException.class, () -> loader.fromLocal(dir));
    }

    @Test
    void builderNullModelFactoryIsRejected() {
        assertThrows(
                NullPointerException.class,
                () -> GGUFTokenizerLoader.createEmptyBuilder().registerModelFactory("test", null));
    }

    @Test
    void builderNullPreTokenizerIsRejected() {
        assertThrows(
                NullPointerException.class,
                () -> GGUFTokenizerLoader.createEmptyBuilder().registerPreTokenizer("test", null));
    }

    @Test
    void builderNullNormalizerIsRejected() {
        assertThrows(
                NullPointerException.class,
                () -> GGUFTokenizerLoader.createEmptyBuilder().registerNormalizer("test", null));
    }

    @Test
    void builderRegisterCustomModelFactory() {
        GGUFTokenizerLoader loader =
                GGUFTokenizerLoader.createEmptyBuilder()
                        .registerModelFactory(
                                "llama",
                                gguf ->
                                        new TokenizationModel() {
                                            @Override
                                            public com.qxotic.toknroll.Vocabulary vocabulary() {
                                                return com.qxotic.toknroll.Toknroll.vocabulary("a");
                                            }

                                            @Override
                                            public void encodeInto(
                                                    CharSequence text,
                                                    int start,
                                                    int end,
                                                    com.qxotic.toknroll.IntSequence.Builder out) {
                                                out.add(0);
                                            }

                                            @Override
                                            public int countTokens(
                                                    CharSequence text, int start, int end) {
                                                return 1;
                                            }

                                            @Override
                                            public int decodeBytesInto(
                                                    com.qxotic.toknroll.IntSequence tokens,
                                                    int idx,
                                                    java.nio.ByteBuffer out) {
                                                return 0;
                                            }

                                            @Override
                                            public float expectedTokensPerChar() {
                                                return 0.5f;
                                            }
                                        })
                        .build();
        assertNotNull(loader);
    }

    @Test
    void builderRegisterCustomPreTokenizer() {
        GGUFTokenizerLoader loader =
                GGUFTokenizerLoader.createEmptyBuilder()
                        .registerPreTokenizer(
                                "default", gguf -> com.qxotic.toknroll.Splitter.identity())
                        .build();
        assertNotNull(loader);
    }

    @Test
    void builderRegisterCustomNormalizer() {
        GGUFTokenizerLoader loader =
                GGUFTokenizerLoader.createEmptyBuilder()
                        .registerNormalizer("default", gguf -> text -> text)
                        .build();
        assertNotNull(loader);
    }
}

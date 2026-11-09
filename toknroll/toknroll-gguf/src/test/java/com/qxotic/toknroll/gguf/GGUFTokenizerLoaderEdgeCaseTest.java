package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
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

    /** A GGUF of another kind says what it is, from its architecture, not which key it lacks. */
    @Test
    void aGgufWithoutATokenizerSaysWhatItIs() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        assertEquals(
                "this GGUF holds no tokenizer (no tokenizer.ggml.model)",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> loader.fromGGUF(Builder.newBuilder().build()))
                        .getMessage());
        assertEquals(
                "this kokoro GGUF holds no tokenizer (no tokenizer.ggml.model)",
                assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        loader.fromGGUF(
                                                Builder.newBuilder()
                                                        .putString("general.architecture", "kokoro")
                                                        .build()))
                        .getMessage());
        assertEquals(
                "this clip GGUF, a vision projector, holds no tokenizer (no tokenizer.ggml.model):"
                        + " its model's GGUF does",
                assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        loader.fromGGUF(
                                                Builder.newBuilder()
                                                        .putString("general.architecture", "clip")
                                                        .build()))
                        .getMessage());
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
                                            public Vocabulary vocabulary() {
                                                return Toknroll.vocabulary("a");
                                            }

                                            @Override
                                            public void encodeInto(
                                                    CharSequence text,
                                                    int start,
                                                    int end,
                                                    IntSequence.Builder out) {
                                                out.add(0);
                                            }

                                            @Override
                                            public int countTokens(
                                                    CharSequence text, int start, int end) {
                                                return 1;
                                            }

                                            @Override
                                            public int decodeBytesInto(
                                                    IntSequence tokens, int idx, ByteBuffer out) {
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
                        .registerPreTokenizer("default", gguf -> Splitter.identity())
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

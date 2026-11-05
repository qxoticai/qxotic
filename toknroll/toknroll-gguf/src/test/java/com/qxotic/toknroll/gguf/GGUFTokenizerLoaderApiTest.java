package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.UnsupportedTokenizerException;
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

    /**
     * Pins builtin coverage of the common llama.cpp pre-tokenizer names (llama-vocab.cpp). Aliasing
     * TO a name only succeeds when that name is registered, so this walks the public API.
     */
    @Test
    void builtinsCoverCommonLlamaCppPreTokenizers() {
        String[] names = {
            "gpt-2",
            "gpt2",
            "granite-docling",
            "exaone4",
            "modern-bert",
            "llama3",
            "llama-v3",
            "llama-bpe",
            "pixtral",
            "smollm3",
            "llama4",
            "glm4",
            "dbrx",
            "smaug-bpe",
            "falcon3",
            "falcon-h1",
            "jina-v5-nano",
            "qwen2",
            "solar-open",
            "hunyuan",
            "grok-2",
            "deepseek-r1-qwen",
            "qwen35",
            "lfm2",
            "tekken",
            "gpt-4o",
            "kanana2",
            "minimax-m2",
            "kimi-k2",
            "gemma4",
            "granite-embed-multi-311m",
            "deepseek-v3",
            "joyai-llm",
            "smollm",
            "command-r",
            "exaone",
            "mellum2",
            "minicpm5",
            "laguna",
            "default",
        };
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        for (String name : names) {
            builder.aliasPreTokenizer("probe-" + name, name);
        }
        assertNotNull(builder.build());
    }

    @Test
    void aliasPreTokenizerWithUnknownTargetFailsAtBuild() {
        // aliases are symbolic: declaring one never throws, but build() validates every target
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("yi", "no-such-scheme");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
        assertTrue(e.getMessage().contains("llama-bpe"));
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

    @Test
    void aFailedFetchNamesTheFileAsAReferenceSpellsIt() {
        // it read "owner/repo@null#x.gguf": a revision only when there is one, a path after a '/'
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        String[][] cases = {
            {null, "ctest-nobody/repo/x.gguf"}, {"v1", "ctest-nobody/repo@v1/x.gguf"}
        };
        for (String[] c : cases) {
            TokenizerLoadException e =
                    assertThrows(
                            TokenizerLoadException.class,
                            () ->
                                    loader.fromHuggingFace(
                                            "ctest-nobody", "repo", c[0], "x.gguf", true, false));
            assertEquals("[huggingface] Failed to load GGUF tokenizer for " + c[1], e.getMessage());
        }
    }

    @Test
    void aModelToknrollDoesNotImplementIsUnsupported() {
        // an unknown pre-tokenizer is too: options_unknown_pre in the C tests, on a whole GGUF
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        GGUF model = Builder.newBuilder().putString("tokenizer.ggml.model", "ctest-model").build();
        UnsupportedTokenizerException e =
                assertThrows(UnsupportedTokenizerException.class, () -> loader.fromGGUF(model));
        assertTrue(e.getMessage().startsWith("Unsupported GGUF tokenizer model 'ctest-model'"));
    }

    @Test
    void aSpellingTwoTokensShareIsRefusedAsLlamaCppRefusesIt() {
        // id("ab") would be either token; llama.cpp asserts as many spellings as tokens
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> loader.fromGGUF(byteLevel("ab", "ab")));
        assertEquals(
                "tokenizer.ggml.tokens[257] spells 'ab', as token 256 does: each token needs a"
                        + " spelling of its own",
                e.getMessage());
        // an empty one is not a spelling: llama.cpp renames it, and loads
        assertNotNull(loader.fromGGUF(byteLevel("", "")));
    }

    /** GPT-2's 256 byte tokens, id = byte, then {@code extra}. */
    private static GGUF byteLevel(String... extra) {
        String[] tokens = new String[256 + extra.length];
        for (int b = 0, shifted = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
            tokens[b] = String.valueOf((char) (printable ? b : 256 + shifted++));
        }
        System.arraycopy(extra, 0, tokens, 256, extra.length);
        return Builder.newBuilder()
                .putString("tokenizer.ggml.model", "gpt2")
                .putString("tokenizer.ggml.pre", "gpt2")
                .putArrayOfString("tokenizer.ggml.tokens", tokens)
                .putArrayOfString("tokenizer.ggml.merges", new String[0])
                .build();
    }
}

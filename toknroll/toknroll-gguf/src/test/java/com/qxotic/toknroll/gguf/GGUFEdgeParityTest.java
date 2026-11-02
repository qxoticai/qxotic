package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DynamicContainer;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestFactory;

/**
 * llama.cpp's own ids on edge-case texts (leading and trailing spaces, empty text, CRLF, byte
 * fallback, emoji, special-token spellings, literal U+2581, long runs), across GGUF families: its
 * vocab-only test files and real models. Both are pure tokenization. The goldens come from {@code
 * scripts/gguf_edge_parity.py}; {@code -Dtoknroll.test.ggufs=DIR} names the folder holding the
 * files by name.
 *
 * <p>Every divergence fails. A known one fails with its cause in the message: llama.cpp matching a
 * user-defined token in plain text, or a control token its gpt-oss workaround renders, which
 * toknroll encodes as text; llama.cpp's gpt-4o, Tekken and Kimi K2 regexes splitting a word at a
 * non-ASCII capital, where the model's own tokenizer and toknroll do not; and llama.cpp cutting
 * Laguna's newlines apart from carriage returns, which its tokenizer.json keeps together.
 */
@Tag("local-external")
class GGUFEdgeParityTest {

    private static final String DIR = System.getProperty("toknroll.test.ggufs");

    /**
     * Files under llama.cpp's rewritten gpt-4o, Tekken and Kimi K2 regexes: every non-ASCII letter
     * both cases.
     */
    private static final Set<String> LLAMA_CPP_CASE_REWRITE =
            Set.of(
                    "gpt-oss-20b-Q8_0.gguf",
                    "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf",
                    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
                    "Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf",
                    "Kimi-K3-UD-IQ1_M-00001-of-00015.gguf");

    @TestFactory
    @SuppressWarnings("unchecked")
    Stream<DynamicContainer> llamaCppParity() throws IOException {
        Assumptions.assumeTrue(DIR != null, "set -Dtoknroll.test.ggufs to the folder of GGUFs");
        Map<String, Object> golden;
        try (InputStream in = getClass().getResourceAsStream("/gguf_edge_parity.json")) {
            golden =
                    (Map<String, Object>)
                            Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
        List<Object> texts = (List<Object>) golden.get("texts");
        return ((List<Object>) golden.get("models"))
                .stream().map(m -> container((Map<String, Object>) m, texts));
    }

    /**
     * One file's cases; a file toknroll cannot load is one failing test with the loader's error.
     */
    @SuppressWarnings("unchecked")
    private static DynamicContainer container(Map<String, Object> model, List<Object> texts) {
        String file = (String) model.get("file");
        Tokenizer tokenizer;
        try {
            tokenizer = load(file);
        } catch (RuntimeException e) {
            return dynamicContainer(
                    file,
                    Stream.of(
                            dynamicTest(
                                    "loads",
                                    () -> {
                                        throw e;
                                    })));
        }
        List<Object> cases = (List<Object>) model.get("cases");
        List<DynamicTest> tests = new ArrayList<>();
        for (int i = 0; i < cases.size(); i++) {
            String text = (String) texts.get(i);
            int[] ids =
                    ((List<Object>) cases.get(i))
                            .stream().mapToInt(id -> ((Number) id).intValue()).toArray();
            tests.add(dynamicTest(printable(text), () -> check(file, tokenizer, text, ids)));
        }
        return dynamicContainer(file, tests);
    }

    private static void check(String file, Tokenizer tokenizer, String text, int[] expected) {
        int[] ids = tokenizer.encode(text).toArray();
        String known = knownCause(file, tokenizer, text, expected);
        assertArrayEquals(
                expected, ids, known == null ? "encode" : "encode, known divergence: " + known);
        assertEquals(expected.length, tokenizer.countTokens(text), "countTokens");
        IntSequence sequence = IntSequence.wrap(ids);
        assertEquals(
                tokenizer.decode(sequence),
                new String(tokenizer.decodeBytes(sequence), StandardCharsets.UTF_8),
                "decode agrees with decodeBytes");
        assertEquals(tokenizer.decodeBytes(sequence).length, tokenizer.countBytes(sequence));
    }

    /** The cause of a divergence already diagnosed, or null; the test fails either way. */
    private static String knownCause(
            String file, Tokenizer tokenizer, String text, int[] expected) {
        if (Arrays.stream(expected)
                .anyMatch(
                        id ->
                                tokenizer
                                        .vocabulary()
                                        .isTokenOfType(id, StandardTokenType.USER_DEFINED))) {
            return "llama.cpp matches a user-defined token in the text; toknroll encodes it as"
                    + " text";
        }
        if (Arrays.stream(expected)
                .anyMatch(
                        id ->
                                tokenizer
                                        .vocabulary()
                                        .isTokenOfType(id, StandardTokenType.CONTROL))) {
            return "llama.cpp matches a control token in the text (its gpt-oss workaround renders"
                    + " Harmony's markers); Hugging Face and toknroll encode it as text";
        }
        if (LLAMA_CPP_CASE_REWRITE.contains(file)
                && text.codePoints().anyMatch(c -> c > 127 && Character.isUpperCase(c))) {
            return "llama.cpp's rewritten regex splits at a non-ASCII capital; the model's own"
                    + " tokenizer and toknroll do not";
        }
        if (file.toLowerCase(Locale.ROOT).startsWith("laguna") && text.contains("\r")) {
            return "llama.cpp's Laguna regex ([^\\n]+|[\\n]+) cuts newlines from carriage returns;"
                    + " Laguna's tokenizer.json keeps CRLF and LF CR together, as toknroll does";
        }
        return null;
    }

    private static Tokenizer load(String file) {
        return GGUFTokenizerLoader.createBuilderWithBuiltins()
                .build()
                .fromLocal(Path.of(DIR, file));
    }

    private static String printable(String text) {
        String shown = text.length() > 40 ? text.substring(0, 40) + "..." : text;
        return "\"" + shown.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t") + "\"";
    }
}

package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.EdgeParityGolden;
import com.qxotic.toknroll.testkit.EdgeParityGolden.Model;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
 * toknroll encodes as text; llama.cpp's gpt-4o, Tekken, Pixtral and Kimi K2 regexes splitting a
 * word at a non-ASCII capital, where the model's own tokenizer and toknroll do not; and llama.cpp
 * cutting Laguna's newlines apart from carriage returns, which its tokenizer.json keeps together.
 * The last two are keyed on the file's {@code tokenizer.ggml.pre} and name only a first difference
 * at such a capital or carriage return.
 */
@Tag("local-external")
class GGUFEdgeParityTest {

    private static final String DIR = System.getProperty("toknroll.test.ggufs");

    /**
     * The pre-tokenizers whose regex llama.cpp rewrites to take every non-ASCII letter both cases.
     */
    private static final Set<String> LLAMA_CPP_CASE_REWRITE =
            Set.of("gpt-4o", "tekken", "pixtral", "kimi-k2");

    @TestFactory
    Stream<DynamicContainer> llamaCppParity() {
        Assumptions.assumeTrue(DIR != null, "set -Dtoknroll.test.ggufs to the folder of GGUFs");
        EdgeParityGolden golden =
                EdgeParityGolden.load(
                        GGUFEdgeParityTest.class,
                        "gguf_edge_parity.json",
                        "toknroll/scripts/gguf_edge_parity.py");
        return golden.models().stream().map(model -> container(model, golden.texts()));
    }

    /**
     * One file's cases; a file toknroll cannot load is one failing test with the loader's error.
     */
    private static DynamicContainer container(Model model, List<String> texts) {
        String file = model.name();
        Tokenizer tokenizer;
        String pre;
        try {
            Path path = Path.of(DIR, file);
            pre = GGUF.read(path).getValueOrDefault(String.class, "tokenizer.ggml.pre", "default");
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromLocal(path);
        } catch (IOException | RuntimeException e) {
            return dynamicContainer(
                    file,
                    Stream.of(
                            dynamicTest(
                                    "loads",
                                    () -> {
                                        throw e;
                                    })));
        }
        List<DynamicTest> tests = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            int[] ids = model.cases().get(i).ids();
            tests.add(
                    dynamicTest(
                            EdgeParityGolden.printable(text),
                            () -> check(pre, tokenizer, text, ids)));
        }
        return dynamicContainer(file, tests);
    }

    private static void check(String pre, Tokenizer tokenizer, String text, int[] expected) {
        int[] ids = tokenizer.encode(text).toArray();
        if (!Arrays.equals(expected, ids)) {
            String known = knownCause(pre, tokenizer, text, expected, ids);
            assertArrayEquals(
                    expected, ids, known == null ? "encode" : "encode, known divergence: " + known);
        }
        EdgeParityGolden.checkCountsAndBytes(
                text,
                expected,
                tokenizer::countTokens,
                t -> tokenizer.decode(IntSequence.wrap(t)),
                t -> tokenizer.decodeBytes(IntSequence.wrap(t)),
                t -> tokenizer.countBytes(IntSequence.wrap(t)));
    }

    /** The cause of a divergence already diagnosed, or null; the test fails either way. */
    private static String knownCause(
            String pre, Tokenizer tokenizer, String text, int[] expected, int[] actual) {
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
        String window = firstDifference(tokenizer, text, expected, actual);
        if (LLAMA_CPP_CASE_REWRITE.contains(pre)
                && window.codePoints().anyMatch(c -> c > 127 && Character.isUpperCase(c))) {
            return "llama.cpp's rewritten regex splits at a non-ASCII capital; the model's own"
                    + " tokenizer and toknroll do not";
        }
        if ("laguna".equals(pre) && window.indexOf('\r') >= 0) {
            return "llama.cpp's Laguna regex ([^\\n]+|[\\n]+) cuts newlines from carriage returns;"
                    + " Laguna's tokenizer.json keeps CRLF and LF CR together, as toknroll does";
        }
        return null;
    }

    /**
     * The text where the ids first differ: the bytes of the longer of the two differing tokens,
     * with a char on each side.
     */
    private static String firstDifference(
            Tokenizer tokenizer, String text, int[] expected, int[] actual) {
        int i = Arrays.mismatch(expected, actual);
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int start = tokenizer.decodeBytes(IntSequence.wrap(Arrays.copyOf(expected, i))).length;
        int width = Math.max(tokenBytes(tokenizer, expected, i), tokenBytes(tokenizer, actual, i));
        int from = Math.max(0, start - 4);
        int to = Math.min(bytes.length, start + width + 4);
        return from >= to ? "" : new String(bytes, from, to - from, StandardCharsets.UTF_8);
    }

    private static int tokenBytes(Tokenizer tokenizer, int[] ids, int i) {
        return i < ids.length ? tokenizer.decodeBytes(IntSequence.of(ids[i])).length : 0;
    }
}

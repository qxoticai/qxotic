package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.EdgeParityGolden;
import com.qxotic.toknroll.testkit.EdgeParityGolden.Expected;
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
 * vocab-only test files and real models, in both modes: plain encode against llama.cpp on the file
 * with its user-defined tokens typed normal (raw text), Specials.encode against llama.cpp with
 * parse_special (every added token matched). The goldens come from {@code
 * scripts/gguf_edge_parity.py}; {@code -Dtoknroll.test.ggufs=DIR} names the folder holding the
 * files by name.
 *
 * <p>Every divergence fails. A known one fails with its cause in the message: llama.cpp matching a
 * control token in raw text, which its gpt-oss workaround renders, or a normal token it retypes
 * control by its name; llama.cpp's gpt-4o, Tekken, Pixtral and Kimi K2 regexes splitting a word at
 * a non-ASCII capital, where the model's own tokenizer and toknroll do not; and llama.cpp cutting
 * Laguna's newlines apart from carriage returns, which its tokenizer.json keeps together. The last
 * two are keyed on the file's {@code tokenizer.ggml.pre} and name only a first difference at such a
 * capital or carriage return.
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
        Specials specials;
        String pre;
        try {
            Path path = Path.of(DIR, file);
            pre = GGUF.read(path).getValueOrDefault(String.class, "tokenizer.ggml.pre", "default");
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromLocal(path);
            specials = Specials.of(tokenizer);
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
            Expected expected = model.cases().get(i);
            String shown = EdgeParityGolden.printable(text);
            tests.add(dynamicTest(shown, () -> check(pre, tokenizer, text, expected.ids())));
            tests.add(
                    dynamicTest(
                            shown + " with specials",
                            () ->
                                    same(
                                            pre,
                                            tokenizer,
                                            text,
                                            expected.specials(),
                                            specials.encode(tokenizer, text).toArray(),
                                            true)));
        }
        return dynamicContainer(file, tests);
    }

    private static void check(String pre, Tokenizer tokenizer, String text, int[] expected) {
        same(pre, tokenizer, text, expected, tokenizer.encode(text).toArray(), false);
        EdgeParityGolden.checkCountsAndBytes(
                text,
                expected,
                tokenizer::countTokens,
                t -> tokenizer.decode(IntSequence.wrap(t)),
                t -> tokenizer.decodeBytes(IntSequence.wrap(t)),
                t -> tokenizer.countBytes(IntSequence.wrap(t)));
    }

    /** Fails unless the ids are llama.cpp's, naming the cause of a divergence already diagnosed. */
    private static void same(
            String pre,
            Tokenizer tokenizer,
            String text,
            int[] expected,
            int[] ids,
            boolean withSpecials) {
        if (!Arrays.equals(expected, ids)) {
            String what = withSpecials ? "Specials.encode" : "encode";
            String known = knownCause(pre, tokenizer, text, expected, ids, withSpecials);
            assertArrayEquals(
                    expected, ids, known == null ? what : what + ", known divergence: " + known);
        }
    }

    /** The cause of a divergence already diagnosed, or null; the test fails either way. */
    private static String knownCause(
            String pre,
            Tokenizer tokenizer,
            String text,
            int[] expected,
            int[] actual,
            boolean withSpecials) {
        if (!withSpecials
                && Arrays.stream(expected)
                        .anyMatch(
                                id ->
                                        tokenizer
                                                .vocabulary()
                                                .isTokenOfType(id, StandardTokenType.CONTROL))) {
            return "llama.cpp matches a control token in raw text (its gpt-oss workaround renders"
                    + " Harmony's markers); Hugging Face and toknroll encode it as text";
        }
        if (withSpecials && matchedWhole(tokenizer, expected, actual)) {
            return "llama.cpp retypes a normal token named like an end-of-generation marker (as"
                    + " </s>) control, so parse_special matches it; the model's tokenizer and"
                    + " toknroll keep it normal";
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
     * Whether llama.cpp's first differing token is a normal one that BPE does not build from its
     * own text: llama.cpp matched it whole, as only a control token is.
     */
    private static boolean matchedWhole(Tokenizer tokenizer, int[] expected, int[] actual) {
        int i = Arrays.mismatch(expected, actual);
        if (i >= expected.length
                || !tokenizer.vocabulary().isTokenOfType(expected[i], StandardTokenType.NORMAL)) {
            return false;
        }
        String token = tokenizer.decode(IntSequence.of(expected[i]));
        return tokenizer.encode(token).length() > 1;
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

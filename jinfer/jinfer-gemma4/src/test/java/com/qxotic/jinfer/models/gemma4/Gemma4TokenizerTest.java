package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link Gemma4Tokenizer} gives the ids of the generic SentencePiece path on Gemma 4's vocabulary:
 * on text built to break its assumptions (space runs, the {@code > </} token that crosses a word
 * start, newlines, byte fallback, surrogates), on random text, on enwik8, and from threads sharing
 * its word cache.
 */
@Tag("integration")
class Gemma4TokenizerTest {

    private static Tokenizer generic;
    private static Gemma4Tokenizer fast;

    @BeforeAll
    static void load() throws IOException {
        GGUF gguf = GGUF.read(TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0"));
        generic = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        fast = new Gemma4Tokenizer(generic);
    }

    private static void assertSameIds(String text) {
        assertArrayEquals(
                generic.encode(text).toArray(),
                fast.encode(text).toArray(),
                () -> "ids differ for: " + visible(text));
    }

    /** The text with surrogates and control chars as \\uXXXX: a lone surrogate hides the rest. */
    private static String visible(String text) {
        StringBuilder out = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (Character.isSurrogate(c) || Character.isISOControl(c)) {
                out.append(String.format("\\u%04X", (int) c));
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    @Test
    void textBuiltToBreakTheSplitEncodesAsTheGenericPath() {
        String longWord = "ab".repeat(100); // past the cached length: merged every time
        for (String text :
                List.of(
                        "",
                        " ",
                        "   ",
                        "\n",
                        "\n\n\n",
                        "a",
                        "Hello world",
                        "Hello  world   again    ",
                        "  leading spaces",
                        "trailing spaces  ",
                        "<b> </b> and > </ and >  </ and x> </y",
                        "line one\nline two\n\n  indented\n",
                        "crlf\r\nline\r\n",
                        "tabs\tand \t mixed \t\t spaces",
                        "emoji \uD83D\uDE00 and \uD83D\uDE00\uD83D\uDE00 runs",
                        "lone \uD800 high and \uDC00 low surrogates",
                        "a literal \u2581 metaspace and \u2581\u2581 two",
                        "中文 日本語 한국어 mixed with English",
                        "combining e\u0301 and n\u0303",
                        longWord,
                        "a " + longWord + " b " + longWord)) {
            assertSameIds(text);
        }
    }

    @Test
    void randomTextEncodesAsTheGenericPath() {
        String alphabet = "aAzZ ' >< /\n\n\r\t.,0123456789éñ\u0301中文日本\uD83D\uDE00\u2581\uD800xy  ";
        Random random = new Random(11);
        for (int i = 0; i < 200_000; i++) {
            StringBuilder text = new StringBuilder();
            for (int k = 0, n = 1 + random.nextInt(40); k < n; k++) {
                text.append(alphabet.charAt(random.nextInt(alphabet.length())));
            }
            if (i % 10 == 0) text.append("<b> </b>  > </  >  </");
            assertSameIds(text.toString());
        }
    }

    @Test
    void enwik8EncodesAsTheGenericPath() throws IOException {
        for (String chunk : enwik8Chunks()) assertSameIds(chunk);
    }

    @Test
    void aRangeEncodesAsItsSubstringAndCountsAsItEncodes() throws IOException {
        for (String chunk : enwik8Chunks()) {
            int from = chunk.length() / 3, to = 2 * chunk.length() / 3;
            IntSequence.Builder out = IntSequence.newBuilder(16);
            fast.encodeInto(chunk, from, to, out);
            int[] expected = generic.encode(chunk.substring(from, to)).toArray();
            assertArrayEquals(expected, out.build().toArray());
            assertEquals(expected.length, fast.countTokens(chunk, from, to));
        }
    }

    @Test
    void threadsSharingTheWordCacheAgreeWithTheGenericPath() throws Exception {
        List<String> chunks = enwik8Chunks();
        List<int[]> expected = new ArrayList<>();
        for (String chunk : chunks) expected.add(generic.encode(chunk).toArray());
        Gemma4Tokenizer shared = new Gemma4Tokenizer(generic);
        try (ExecutorService pool = Executors.newFixedThreadPool(8)) {
            List<Future<?>> runs = new ArrayList<>();
            for (int t = 0; t < 8; t++) {
                List<Integer> order = new ArrayList<>();
                for (int i = 0; i < chunks.size(); i++) order.add(i);
                Collections.shuffle(order, new Random(t));
                runs.add(
                        pool.submit(
                                () -> {
                                    for (int i : order) {
                                        assertArrayEquals(
                                                expected.get(i),
                                                shared.encode(chunks.get(i)).toArray(),
                                                "chunk " + i);
                                    }
                                }));
            }
            for (Future<?> run : runs) run.get();
        }
    }

    /** The enwik8 chunks of test-fixtures/golden/enwik8/chunks.json, found up from here. */
    private static List<String> enwik8Chunks() throws IOException {
        Path file = null;
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            Path candidate = dir.resolve("test-fixtures/golden/enwik8/chunks.json");
            if (Files.isRegularFile(candidate)) {
                file = candidate;
                break;
            }
        }
        assumeTrue(file != null, "test-fixtures/golden/enwik8/chunks.json not found");
        List<String> chunks = new ArrayList<>();
        for (Object chunk : Json.parseList(Files.readString(file))) {
            chunks.add((String) ((Map<?, ?>) chunk).get("text"));
        }
        return chunks;
    }
}

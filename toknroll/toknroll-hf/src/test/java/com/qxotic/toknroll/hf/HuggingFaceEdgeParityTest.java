package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicContainer;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestFactory;

/**
 * Hugging Face's own answers on edge-case texts (leading and trailing spaces, empty text, CRLF,
 * byte fallback, emoji, special-token spellings, literal U+2581, long runs), across tokenizer.json
 * families: ids, text, bytes and counts. The goldens come from {@code scripts/hf_edge_parity.py}.
 * Every divergence fails. A known one, Hugging Face matching a non-special added token that
 * toknroll encodes as text, fails with its cause in the message.
 *
 * <p>Tokenizers load from the hub, or with {@code -Dtoknroll.test.tokenizers=DIR} from {@code
 * DIR/owner__repo/tokenizer.json}, the files the goldens were made from.
 */
@Tag("local-external")
@Tag("network")
class HuggingFaceEdgeParityTest {

    private static final String LOCAL = System.getProperty("toknroll.test.tokenizers");

    @TestFactory
    @SuppressWarnings("unchecked")
    Stream<DynamicContainer> hfParity() throws IOException {
        Map<String, Object> golden;
        try (InputStream in = getClass().getResourceAsStream("/hf_edge_parity.json")) {
            golden =
                    (Map<String, Object>)
                            Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
        List<Object> texts = (List<Object>) golden.get("texts");
        return ((List<Object>) golden.get("models"))
                .stream()
                        .map(
                                m -> {
                                    Map<String, Object> model = (Map<String, Object>) m;
                                    String ref = (String) model.get("ref");
                                    Supplier<Tokenizer> tokenizer = memoize(() -> load(ref));
                                    List<Object> cases = (List<Object>) model.get("cases");
                                    List<DynamicTest> tests = new ArrayList<>();
                                    for (int i = 0; i < cases.size(); i++) {
                                        String text = (String) texts.get(i);
                                        Map<String, Object> expected =
                                                (Map<String, Object>) cases.get(i);
                                        tests.add(
                                                dynamicTest(
                                                        printable(text),
                                                        () ->
                                                                check(
                                                                        tokenizer.get(),
                                                                        text,
                                                                        expected)));
                                    }
                                    return dynamicContainer(ref, tests);
                                });
    }

    @SuppressWarnings("unchecked")
    private static void check(Tokenizer tokenizer, String text, Map<String, Object> expected) {
        int[] ids =
                ((List<Object>) expected.get("ids"))
                        .stream().mapToInt(id -> ((Number) id).intValue()).toArray();
        String decoded = (String) expected.get("decoded");
        String known = (String) expected.get("known");
        assertArrayEquals(
                ids,
                tokenizer.encode(text).toArray(),
                known == null ? "encode" : "encode, known divergence: " + known);
        assertEquals(ids.length, tokenizer.countTokens(text), "countTokens");
        IntSequence sequence = IntSequence.wrap(ids);
        assertEquals(decoded, tokenizer.decode(sequence), "decode");
        assertEquals(
                decoded,
                new String(tokenizer.decodeBytes(sequence), StandardCharsets.UTF_8),
                "decodeBytes");
        assertEquals(tokenizer.decodeBytes(sequence).length, tokenizer.countBytes(sequence));
    }

    private static Tokenizer load(String ref) {
        if (LOCAL != null) {
            Path file = Path.of(LOCAL, ref.replace("/", "__"), "tokenizer.json");
            if (Files.exists(file)) {
                return HuggingFaceTokenizerLoader.fromLocal(file);
            }
        }
        String[] ownerRepo = ref.split("/", 2);
        return HuggingFaceTokenizerLoader.fromHuggingFace(ownerRepo[0], ownerRepo[1]);
    }

    private static <T> Supplier<T> memoize(Supplier<T> supplier) {
        return new Supplier<>() {
            private T value;

            @Override
            public synchronized T get() {
                if (value == null) {
                    value = supplier.get();
                }
                return value;
            }
        };
    }

    private static String printable(String text) {
        String shown = text.length() > 40 ? text.substring(0, 40) + "..." : text;
        return "\"" + shown.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t") + "\"";
    }
}

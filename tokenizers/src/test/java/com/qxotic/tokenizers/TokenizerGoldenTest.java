package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.json.Json;
import com.qxotic.tokenizers.testkit.TiktokenFixtures;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerGoldenTest {

    @ParameterizedTest(name = "golden {0}/{1}")
    @MethodSource("goldenCases")
    void goldenEncodingMatchesExpected(
            String encoding,
            String caseId,
            String text,
            int[] expectedTokens,
            String expectedDecoded,
            byte[] expectedDecodedBytes,
            int expectedTokenCount) {

        Tokenizer tokenizer = TiktokenFixtures.createJtokkitTokenizer(encoding);
        IntSequence tokens = tokenizer.encode(text);

        assertArrayEquals(expectedTokens, tokens.toArray(), encoding + "/" + caseId + " tokens");
        assertEquals(
                expectedDecoded, tokenizer.decode(tokens), encoding + "/" + caseId + " decoded");
        assertArrayEquals(
                expectedDecodedBytes,
                tokenizer.decodeBytes(tokens),
                encoding + "/" + caseId + " decoded bytes");
        assertEquals(
                expectedTokenCount,
                tokenizer.countTokens(text),
                encoding + "/" + caseId + " count");
    }

    static Stream<Arguments> goldenCases() {
        Map<String, Object> root = loadGoldenJson();
        List<Arguments> args = new ArrayList<>();

        // Keep this deterministic and fast: validate all r50k cases + representative cl100k/o200k
        // cases.
        collectCases(args, root, "r50k_base", true);
        collectCases(args, root, "cl100k_base", false);
        collectCases(args, root, "o200k_base", false);

        return args.stream();
    }

    @SuppressWarnings("unchecked")
    private static void collectCases(
            List<Arguments> out, Map<String, Object> root, String encoding, boolean allCases) {
        Object encodingObj = root.get(encoding);
        if (!(encodingObj instanceof Map<?, ?> encodingMap)) {
            return;
        }

        int collected = 0;
        for (Map.Entry<String, Object> entry : ((Map<String, Object>) encodingMap).entrySet()) {
            String caseId = entry.getKey();
            if (!(entry.getValue() instanceof Map<?, ?> caseMapRaw)) {
                continue;
            }
            Map<String, Object> c = (Map<String, Object>) caseMapRaw;

            String text = asString(c.get("text"));
            String decoded = asString(c.get("decoded"));
            List<Object> tokenValues = asList(c.get("tokens"));
            List<Object> decodedByteValues = asList(c.get("decoded_bytes"));
            if (text == null
                    || decoded == null
                    || tokenValues == null
                    || decodedByteValues == null) {
                continue;
            }

            String inputText = chooseInputText(text, decoded);
            int[] tokens = toIntArray(tokenValues);
            byte[] decodedBytes = toByteArray(decodedByteValues);
            int tokenCount =
                    c.get("token_count") instanceof Number tokenCountValue
                            ? tokenCountValue.intValue()
                            : tokens.length;

            out.add(
                    Arguments.of(
                            encoding,
                            caseId,
                            inputText,
                            tokens,
                            decoded,
                            decodedBytes,
                            tokenCount));
            collected++;
            if (!allCases && collected >= 25) {
                break;
            }
        }
    }

    private static String asString(Object value) {
        return value instanceof String s ? s : null;
    }

    private static String chooseInputText(String text, String decoded) {
        if (text == null) {
            return decoded;
        }
        if (decoded == null) {
            return text;
        }
        if (isAllQuestionMarks(text) && !isAllQuestionMarks(decoded)) {
            return decoded;
        }
        return text;
    }

    private static boolean isAllQuestionMarks(String value) {
        if (value.isEmpty()) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            if (value.charAt(i) != '?') {
                return false;
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object value) {
        return value instanceof List<?> list ? (List<Object>) list : null;
    }

    private static int[] toIntArray(List<Object> values) {
        int[] arr = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            arr[i] = ((Number) values.get(i)).intValue();
        }
        return arr;
    }

    private static byte[] toByteArray(List<Object> values) {
        byte[] arr = new byte[values.size()];
        for (int i = 0; i < values.size(); i++) {
            arr[i] = ((Number) values.get(i)).byteValue();
        }
        return arr;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> loadGoldenJson() {
        try (InputStream is =
                TokenizerGoldenTest.class
                        .getClassLoader()
                        .getResourceAsStream("ground_truth_tokens.json")) {
            if (is == null) {
                throw new IllegalStateException("Missing ground_truth_tokens.json");
            }
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            return (Map<String, Object>) Json.parse(json);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load golden fixture", e);
        }
    }
}

package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ByteEncoding;
import com.qxotic.toknroll.impl.Decoder;
import com.qxotic.toknroll.impl.RegexSplitter;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GGUFTokenizerComponentsTest {

    @Test
    void qwenAndLlamaPatternsSplitNumbersDifferently() {
        Splitter qwen = RegexSplitter.create(ModelTextSplitters.QWEN2_PATTERN);
        Splitter llama = RegexSplitter.create(ModelTextSplitters.LLAMA3_PATTERN);

        String text = "v 1234";
        List<String> qwenTokens =
                qwen.splitAllToListEagerly(text).stream().map(CharSequence::toString).toList();
        List<String> llamaTokens =
                llama.splitAllToListEagerly(text).stream().map(CharSequence::toString).toList();

        // Qwen pattern uses single digits; Llama/Tekken uses 1..3 digit chunks.
        assertTrue(qwenTokens.stream().anyMatch(t -> t.contains("1")));
        assertTrue(qwenTokens.stream().anyMatch(t -> t.contains("4")));
        assertTrue(llamaTokens.stream().anyMatch(t -> t.contains("123") || t.contains(" 123")));
    }

    @Test
    void tekkenLlamaPatternHandlesContractions() {
        Splitter splitter = RegexSplitter.create(ModelTextSplitters.LLAMA3_PATTERN);
        List<String> tokens =
                splitter.splitAllToListEagerly("I'm ready").stream()
                        .map(CharSequence::toString)
                        .toList();
        assertTrue(tokens.stream().anyMatch(t -> t.contains("'m")));
    }

    @Test
    void metaspaceDecoderDecodesLlamaStyleTokens() {
        Vocabulary vocab = simpleVocabulary(Map.of("▁Hello", 1, "▁world", 2));
        Decoder decoder = Decoder.metaspace('▁', true);
        String text = decoder.decode(IntSequence.of(1, 2), vocab);
        assertEquals(" Hello world", text);
    }

    @Test
    void byteLevelDecoderRoundTripsUtf8() {
        String input = "Hello 🌍";
        String encoded = ByteEncoding.bytesToString(input.getBytes(StandardCharsets.UTF_8));
        Vocabulary vocab = simpleVocabulary(Map.of(encoded, 1));

        String decoded = Decoder.byteLevel().decode(IntSequence.of(1), vocab);
        assertEquals(input, decoded);
    }

    @Test
    void preTokenizerAdapterUsesModelSplitter() {
        Splitter splitter = ModelTextSplitters.createSplitter("qwen3");

        String input = "Hello, world";
        List<CharSequence> tokens = splitter.splitAllToListEagerly(input);

        assertTrue(tokens.size() >= 2);
        String rebuilt =
                tokens.stream()
                        .map(CharSequence::toString)
                        .collect(java.util.stream.Collectors.joining());
        assertEquals(input, rebuilt);
    }

    private static Vocabulary simpleVocabulary(Map<String, Integer> tokenToId) {
        Map<Integer, String> idToToken =
                tokenToId.entrySet().stream()
                        .collect(
                                java.util.stream.Collectors.toMap(
                                        Map.Entry::getValue, Map.Entry::getKey));
        return new Vocabulary() {
            @Override
            public String token(int id) {
                return idToToken.get(id);
            }

            @Override
            public int id(String token) {
                return tokenToId.getOrDefault(token, -1);
            }

            @Override
            public int size() {
                return tokenToId.size();
            }

            @Override
            public boolean contains(int id) {
                return idToToken.containsKey(id);
            }

            @Override
            public boolean contains(String text) {
                return tokenToId.containsKey(text);
            }

            @Override
            public Iterator<Map.Entry<String, Integer>> iterator() {
                return tokenToId.entrySet().iterator();
            }
        };
    }
}

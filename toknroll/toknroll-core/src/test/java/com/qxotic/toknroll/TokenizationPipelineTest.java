package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Tokenization Pipeline Tests")
class TokenizationPipelineTest {

    @Test
    void splitterIdentityReturnsSingleChunk() {
        Splitter splitter = Splitter.identity();
        List<CharSequence> chunks = splitter.splitAllToListEagerly("Hello world");

        assertEquals(1, chunks.size());
        assertEquals("Hello world", chunks.get(0).toString());
    }

    @Test
    void splitterSequenceComposesStages() {
        Splitter spaceSplitter =
                (text, startInclusive, endExclusive, consumer) -> {
                    int start = startInclusive;
                    for (int i = startInclusive; i <= endExclusive; i++) {
                        if (i == endExclusive || text.charAt(i) == ' ') {
                            if (start < i) {
                                consumer.accept(text, start, i);
                            }
                            start = i + 1;
                        }
                    }
                };
        Splitter splitter = Splitter.sequence(spaceSplitter);

        List<CharSequence> chunks = splitter.splitAllToListEagerly("Hello world test");
        assertEquals(
                List.of("Hello", "world", "test"),
                chunks.stream().map(Object::toString).collect(Collectors.toList()));
    }

    @Test
    void pipelineBuilderRejectsNullModel() {
        assertThrows(NullPointerException.class, () -> TokenizationPipeline.builder(null));
    }

    @Test
    void pipelineBuilderRejectsNullSplitter() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello", 1));
        assertThrows(
                NullPointerException.class,
                () -> TokenizationPipeline.builder(model).splitter(null));
    }

    @Test
    void pipelineMinimalConfigWorks() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello", 1, "world", 2));
        Tokenizer pipeline = TokenizationPipeline.builder(model).build();

        IntSequence tokens = pipeline.encode("hello");
        assertEquals(1, tokens.length());
        assertEquals(1, tokens.intAt(0));
        assertSame(model.vocabulary(), pipeline.vocabulary());
    }

    @Test
    void pipelineCountTokensMatchesEncodeOnSlice() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello", 1, "world", 2));
        Tokenizer pipeline =
                TokenizationPipeline.builder(model)
                        .splitter(Splitter.sequence(spaceOnlySplitter()))
                        .build();

        String text = "hello world";
        assertEquals(pipeline.encode(text).length(), pipeline.countTokens(text));
        assertEquals(
                pipeline.encode(text.subSequence(0, 5)).length(), pipeline.countTokens(text, 0, 5));
    }

    @Test
    void pipelineAppliesNormalizerBeforeSplitter() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello", 10, "world", 11));
        Tokenizer pipeline =
                TokenizationPipeline.builder(model)
                        .normalizer(Normalizer.lowercase())
                        .splitter(Splitter.sequence(spaceOnlySplitter()))
                        .build();

        assertArrayEquals(new int[] {10, 11}, pipeline.encodeToArray("HELLO WORLD"));
    }

    @Test
    void pipelineCountAndArrayConvenienceMethodsMatchCoreMethods() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello", 200, "world", 201));
        Tokenizer pipeline =
                TokenizationPipeline.builder(model)
                        .splitter(Splitter.sequence(spaceOnlySplitter()))
                        .build();

        String text = "hello world";
        IntSequence tokens = pipeline.encode(text);
        int[] tokenArray = pipeline.encodeToArray(text);

        assertEquals(tokens.length(), pipeline.countTokens(text));
        assertArrayEquals(tokens.toArray(), tokenArray);
        assertEquals(pipeline.decode(tokens), pipeline.decode(tokenArray));
        assertArrayEquals(pipeline.decodeBytes(tokens), pipeline.decodeBytes(tokenArray));
    }

    @Test
    void builderGettersExposeCurrentConfiguration() {
        TokenizationModel model = createWholeChunkModel(Map.of("x", 1));
        TokenizationPipeline.Builder builder = TokenizationPipeline.builder(model);

        assertSame(model, builder.model());
        assertTrue(builder.splitter().isEmpty());

        Splitter splitter = Splitter.identity();
        builder.splitter(splitter);
        assertEquals(Optional.of(splitter), builder.splitter());
    }

    @Test
    void pipelineWithNoStagesUsesModelOutput() {
        TokenizationModel model = createWholeChunkModel(Map.of("hello world", 7));
        Tokenizer pipeline = TokenizationPipeline.builder(model).build();

        assertArrayEquals(
                model.encodeToArray("hello world"), pipeline.encodeToArray("hello world"));
        assertFalse(((TokenizationPipeline) pipeline).splitter().isPresent());
    }

    private Vocabulary createSimpleVocabulary(String[] tokens, int[] ids) {
        Map<String, Integer> tokenToId = new HashMap<>();
        Map<Integer, String> idToToken = new HashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.put(tokens[i], ids[i]);
            idToToken.put(ids[i], tokens[i]);
        }

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
                return tokens.length;
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

    private Vocabulary createVocabularyFromMap(Map<String, Integer> tokenMap) {
        Map<Integer, String> idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> entry : tokenMap.entrySet()) {
            idToToken.put(entry.getValue(), entry.getKey());
        }

        return new Vocabulary() {
            @Override
            public String token(int id) {
                return idToToken.get(id);
            }

            @Override
            public int id(String token) {
                return tokenMap.getOrDefault(token, -1);
            }

            @Override
            public int size() {
                return tokenMap.size();
            }

            @Override
            public boolean contains(int id) {
                return idToToken.containsKey(id);
            }

            @Override
            public boolean contains(String text) {
                return tokenMap.containsKey(text);
            }

            @Override
            public Iterator<Map.Entry<String, Integer>> iterator() {
                return tokenMap.entrySet().iterator();
            }
        };
    }

    private TokenizationModel createWholeChunkModel(Map<String, Integer> tokens) {
        Vocabulary vocab = createVocabularyFromMap(tokens);
        return new TokenizationModel() {
            @Override
            public Vocabulary vocabulary() {
                return vocab;
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                String slice = text.subSequence(startInclusive, endExclusive).toString();
                Integer id = tokens.get(slice);
                if (id != null) {
                    out.add(id);
                }
            }

            @Override
            public String decode(IntSequence tokensSeq) {
                List<String> out = new ArrayList<>();
                for (int i = 0; i < tokensSeq.length(); i++) {
                    String token = vocab.token(tokensSeq.intAt(i));
                    if (token != null) {
                        out.add(token);
                    }
                }
                return String.join(" ", out);
            }

            @Override
            public byte[] decodeBytes(IntSequence tokensSeq) {
                return decode(tokensSeq).getBytes(StandardCharsets.UTF_8);
            }

            @Override
            public int decodeBytesInto(IntSequence tokensSeq, int tokenStartIndex, ByteBuffer out) {
                int length = tokensSeq.length();
                if (tokenStartIndex < 0 || tokenStartIndex > length) {
                    throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
                }
                if (tokenStartIndex == length) {
                    return 0;
                }
                byte[] bytes = decodeBytes(tokensSeq.subSequence(tokenStartIndex, length));
                if (bytes.length > out.remaining()) {
                    throw new IllegalArgumentException("Not enough output space");
                }
                out.put(bytes);
                return length - tokenStartIndex;
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                return encode(text.subSequence(startInclusive, endExclusive)).length();
            }

            @Override
            public int countBytes(IntSequence tokensSeq) {
                return decodeBytes(tokensSeq).length;
            }
        };
    }

    private static Splitter spaceOnlySplitter() {
        return (text, startInclusive, endExclusive, consumer) -> {
            int start = startInclusive;
            for (int i = startInclusive; i <= endExclusive; i++) {
                if (i == endExclusive || text.charAt(i) == ' ') {
                    if (start < i) {
                        consumer.accept(text, start, i);
                    }
                    start = i + 1;
                }
            }
        };
    }
}

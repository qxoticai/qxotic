package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.advanced.TokenizationPipeline;
import java.nio.ByteBuffer;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
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
    void pipelineBuilderRequiresBaseTokenizer() {
        IllegalStateException none =
                assertThrows(
                        IllegalStateException.class, () -> TokenizationPipeline.builder().build());
        assertTrue(none.getMessage().contains("baseTokenizer"));
    }

    @Test
    void pipelineBuilderRejectsNullComponentsEarly() {
        TokenizationPipeline.Builder builder = TokenizationPipeline.builder();
        assertThrows(NullPointerException.class, () -> builder.baseTokenizer(null));
        assertThrows(NullPointerException.class, () -> builder.normalizer(null));
        assertThrows(NullPointerException.class, () -> builder.splitter(null));
        assertThrows(NullPointerException.class, () -> builder.postProcessor(null));
    }

    @Test
    void pipelineMinimalConfigWorks() {
        Tokenizer base = createWholeChunkTokenizer(Map.of("hello", 1, "world", 2));
        TokenizationPipeline pipeline = TokenizationPipeline.builder(base).build();

        IntSequence tokens = pipeline.encode("hello");
        assertEquals(1, tokens.length());
        assertEquals(1, tokens.intAt(0));
        assertSame(base.vocabulary(), pipeline.vocabulary());
    }

    @Test
    void pipelineAppliesNormalizerSplitterAndPostProcessor() {
        Tokenizer base = createWholeChunkTokenizer(Map.of("é", 10, "world", 11, "<eos>", 99));
        Function<IntSequence, IntSequence> eosAppender =
                tokens -> {
                    IntSequence.Builder out = IntSequence.newBuilder();
                    out.addAll(tokens);
                    out.add(99);
                    return out.build();
                };

        TokenizationPipeline pipeline =
                TokenizationPipeline.builder(base)
                        .normalizer(Normalizer.unicode(Form.NFKC))
                        .splitter(Splitter.sequence(spaceOnlySplitter()))
                        .postProcessor(eosAppender)
                        .build();

        IntSequence tokens = pipeline.encode("e\u0301 world");
        assertArrayEquals(new int[] {10, 11, 99}, tokens.toArray());
    }

    @Test
    void pipelineCountAndArrayConvenienceMethodsMatchCoreMethods() {
        Tokenizer base = createWholeChunkTokenizer(Map.of("hello", 200, "world", 201));
        Tokenizer pipeline =
                TokenizationPipeline.builder(base)
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
        Tokenizer base = createWholeChunkTokenizer(Map.of("x", 1));
        TokenizationPipeline.Builder builder = TokenizationPipeline.builder().baseTokenizer(base);

        assertEquals(Optional.of(base), builder.baseTokenizer());
        assertTrue(builder.normalizer().isEmpty());
        assertTrue(builder.splitter().isEmpty());
        assertTrue(builder.postProcessor().isEmpty());

        Normalizer normalizer = Normalizer.lowercase();
        Splitter splitter = Splitter.identity();
        Function<IntSequence, IntSequence> post = Function.identity();

        builder.normalizer(normalizer).splitter(splitter).postProcessor(post);
        assertEquals(Optional.of(normalizer), builder.normalizer());
        assertEquals(Optional.of(splitter), builder.splitter());
        assertEquals(Optional.of(post), builder.postProcessor());
    }

    @Test
    void normalizerReplacementAndExplicitSequenceCompositionWork() {
        Tokenizer base = createWholeChunkTokenizer(Map.of("é", 5));
        TokenizationPipeline.Builder builder = TokenizationPipeline.builder(base);

        builder.normalizer(Normalizer.lowercase());
        Normalizer append = Normalizer.unicode(Form.NFC);
        Normalizer combined =
                builder.normalizer()
                        .map(current -> Normalizer.sequence(current, append))
                        .orElse(append);
        builder.normalizer(combined);

        IntSequence tokens = builder.build().encode("E\u0301");
        assertArrayEquals(new int[] {5}, tokens.toArray());
    }

    @Test
    void pipelineWithNoStagesUsesBaseTokenizerOutput() {
        Tokenizer base = createWholeChunkTokenizer(Map.of("hello world", 7));
        TokenizationPipeline pipeline = TokenizationPipeline.builder(base).build();

        assertArrayEquals(base.encodeToArray("hello world"), pipeline.encodeToArray("hello world"));
        assertFalse(pipeline.normalizer().isPresent());
        assertFalse(pipeline.splitter().isPresent());
        assertFalse(pipeline.postProcessor().isPresent());
    }

    private Vocabulary createSimpleVocabulary(String[] tokens, int[] ids) {
        Map<String, Integer> tokenToId = new java.util.HashMap<>();
        Map<Integer, String> idToToken = new java.util.HashMap<>();
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
        Map<Integer, String> idToToken = new java.util.HashMap<>();
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

    private Tokenizer createWholeChunkTokenizer(Map<String, Integer> tokens) {
        Vocabulary vocab = createVocabularyFromMap(tokens);
        return new Tokenizer() {
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
                return decode(tokensSeq).getBytes(java.nio.charset.StandardCharsets.UTF_8);
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
            public int countTokens(CharSequence text) {
                return encode(text).length();
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

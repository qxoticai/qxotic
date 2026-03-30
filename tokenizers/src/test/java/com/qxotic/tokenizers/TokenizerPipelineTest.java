package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.advanced.Decoder;
import com.qxotic.tokenizers.advanced.Encoder;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.advanced.TokenizerPipeline;
import java.text.Normalizer.Form;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Tokenizer Pipeline Tests")
class TokenizerPipelineTest {

    @Test
    void splitterIdentityReturnsSingleChunk() {
        Splitter splitter = Splitter.identity();
        List<CharSequence> chunks = splitter.split("Hello world");

        assertEquals(1, chunks.size());
        assertEquals("Hello world", chunks.get(0).toString());
    }

    @Test
    void splitterSequenceComposesStages() {
        Splitter spaceSplitter = text -> List.of(text.toString().split(" "));
        Splitter splitter = Splitter.sequence(spaceSplitter);

        List<CharSequence> chunks = splitter.split("Hello world test");
        assertEquals(
                List.of("Hello", "world", "test"), chunks.stream().map(Object::toString).toList());
    }

    @Test
    void decoderCanonicalConcatenatesTokenStrings() {
        Vocabulary vocab =
                createSimpleVocabulary(
                        new String[] {"[UNK]", "Hello", " ", "world"}, new int[] {0, 1, 2, 3});
        Decoder decoder = Decoder.canonical();

        String result = decoder.decode(IntSequence.of(1, 2, 3), vocab);
        assertEquals("Hello world", result);
    }

    @Test
    void pipelineBuilderRequiresEncoderAndVocabulary() {
        assertThrows(NullPointerException.class, () -> new TokenizerPipeline.Builder().build());
        assertThrows(
                NullPointerException.class,
                () ->
                        new TokenizerPipeline.Builder()
                                .encoder(chunk -> IntSequence.empty())
                                .build());
        assertThrows(
                NullPointerException.class,
                () ->
                        new TokenizerPipeline.Builder()
                                .vocabulary(
                                        createSimpleVocabulary(new String[] {"a"}, new int[] {0}))
                                .build());
    }

    @Test
    void pipelineMinimalConfigWorks() {
        Vocabulary vocab =
                createSimpleVocabulary(
                        new String[] {"[UNK]", "Hello", "world", "!"}, new int[] {0, 1, 2, 3});
        Encoder encoder =
                chunk -> {
                    IntSequence.Builder ids = IntSequence.newBuilder();
                    for (String word : chunk.toString().split(" ")) {
                        int id = vocab.id(word);
                        ids.add(id >= 0 ? id : 0);
                    }
                    return ids.build();
                };

        TokenizerPipeline pipeline =
                new TokenizerPipeline.Builder().encoder(encoder).vocabulary(vocab).build();

        IntSequence tokens = pipeline.encode("Hello world");
        assertEquals(2, tokens.length());
        assertEquals(1, tokens.intAt(0));
        assertEquals(2, tokens.intAt(1));
    }

    @Test
    void pipelineEndToEndWorks() {
        Map<String, Integer> tokenMap = Map.of("hello", 200, "world", 201, "[UNK]", 0);
        Vocabulary vocab = createVocabularyFromMap(tokenMap);
        Encoder encoder =
                chunk -> {
                    IntSequence.Builder ids = IntSequence.newBuilder();
                    for (String word : chunk.toString().split(" ")) {
                        ids.add(tokenMap.getOrDefault(word.toLowerCase(), 0));
                    }
                    return ids.build();
                };

        TokenizerPipeline pipeline =
                new TokenizerPipeline.Builder()
                        .normalizer(Normalizer.unicode(Form.NFKC))
                        .splitter(Splitter.identity())
                        .encoder(encoder)
                        .vocabulary(vocab)
                        .decoder(Decoder.canonical())
                        .build();

        IntSequence tokens = pipeline.encode("Hello World");
        assertEquals(2, tokens.length());
        assertEquals(200, tokens.intAt(0));
        assertEquals(201, tokens.intAt(1));

        String decoded = pipeline.decode(tokens);
        assertNotNull(decoded);
        assertTrue(decoded.contains("hello") || decoded.contains("world"));
    }

    @Test
    void pipelineComponentsAccessible() {
        Normalizer normalizer = Normalizer.identity();
        Splitter splitter = Splitter.identity();
        Vocabulary vocab = createSimpleVocabulary(new String[] {"[UNK]"}, new int[] {0});
        Encoder encoder = chunk -> IntSequence.empty();
        Decoder decoder = Decoder.canonical();

        TokenizerPipeline pipeline =
                new TokenizerPipeline.Builder()
                        .normalizer(normalizer)
                        .splitter(splitter)
                        .encoder(encoder)
                        .vocabulary(vocab)
                        .decoder(decoder)
                        .build();

        assertSame(normalizer, pipeline.normalizer());
        assertSame(splitter, pipeline.splitter());
        assertSame(encoder, pipeline.encoder());
        assertSame(decoder, pipeline.decoder());
        assertSame(vocab, pipeline.vocabulary());
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
}

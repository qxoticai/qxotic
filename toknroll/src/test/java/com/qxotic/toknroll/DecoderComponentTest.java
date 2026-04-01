package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.Decoder;
import com.qxotic.toknroll.advanced.SymbolCodec;
import java.util.Iterator;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DecoderComponentTest {

    @Test
    void fromCodecUsesIdentityCodec() {
        Vocabulary vocab = simpleVocabulary(Map.of("hello", 1, " world", 2));
        Decoder decoder = Decoder.fromCodec(SymbolCodec.IDENTITY);
        assertEquals("hello world", decoder.decode(IntSequence.of(1, 2), vocab));
    }

    @Test
    void byteLevelDecodesUtf8Content() {
        String input = "Hi 🌍";
        String encoded = SymbolCodec.BYTE_LEVEL.encodeText(input);
        Vocabulary vocab = simpleVocabulary(Map.of(encoded, 5));

        String decoded = Decoder.byteLevel().decode(IntSequence.of(5), vocab);
        assertEquals(input, decoded);
    }

    @Test
    void metaspaceRespectsPrependScheme() {
        Vocabulary vocab = simpleVocabulary(Map.of("▁Hello", 1, "▁world", 2));

        String noPrepend = Decoder.metaspace('▁', false).decode(IntSequence.of(1, 2), vocab);
        String withPrepend = Decoder.metaspace('▁', true).decode(IntSequence.of(1, 2), vocab);

        assertEquals("▁Hello world", noPrepend);
        assertEquals(" Hello world", withPrepend);
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

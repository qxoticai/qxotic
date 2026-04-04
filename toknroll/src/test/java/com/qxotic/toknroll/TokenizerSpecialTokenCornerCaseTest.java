package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerSpecialTokenCornerCaseTest {

    @ParameterizedTest(name = "special token text is encoded as regular text {0}")
    @MethodSource("tokenizers")
    void specialTokenTextIsEncodedAsRegularText(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        Map<String, Integer> specials = fixture(namedTokenizer).specialTokens();
        String[] texts = {"<|endoftext|>", "Hello <|endoftext|> world", "Start <|endoftext|> end"};
        for (String text : texts) {
            IntSequence encoded = tokenizer.encode(text);
            assertEquals(text, tokenizer.decode(encoded), namedTokenizer.name() + " round-trip " + text);
            for (Integer specialId : specials.values()) {
                assertTrue(
                        !containsId(encoded, specialId),
                        namedTokenizer.name() + " should not emit special id " + specialId + " for text " + text);
            }
        }
    }

    private static boolean containsId(IntSequence encoded, int tokenId) {
        for (int i = 0; i < encoded.length(); i++) {
            if (encoded.intAt(i) == tokenId) {
                return true;
            }
        }
        return false;
    }

    @ParameterizedTest(name = "special token decode {0}")
    @MethodSource("tokenizers")
    void specialTokenIdCanBeDecoded(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        Map<String, Integer> specials = fixture(namedTokenizer).specialTokens();

        for (Map.Entry<String, Integer> special : specials.entrySet()) {
            IntSequence token = IntSequence.of(special.getValue());
            assertTrue(
                    tokenizer.vocabulary().contains(special.getValue()),
                    namedTokenizer.name() + " has special id");
            assertEquals(
                    special.getKey(),
                    tokenizer.decode(token),
                    namedTokenizer.name() + " special decode");
            assertArrayEquals(
                    special.getKey().getBytes(StandardCharsets.UTF_8),
                    tokenizer.decodeBytes(token),
                    namedTokenizer.name() + " special decode bytes");
        }
    }

    @ParameterizedTest(name = "special token in vocab {0}")
    @MethodSource("tokenizers")
    void specialTokensExistInVocabulary(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Vocabulary vocab = namedTokenizer.tokenizer().vocabulary();
        Map<String, Integer> specials = fixture(namedTokenizer).specialTokens();

        for (Map.Entry<String, Integer> special : specials.entrySet()) {
            assertTrue(
                    vocab.contains(special.getValue()),
                    namedTokenizer.name() + " contains special id");
            assertEquals(
                    special.getKey(),
                    vocab.token(special.getValue()),
                    namedTokenizer.name() + " token text");
        }
    }

    private static TiktokenFixtures.EncodingFixture fixture(
            TiktokenFixtures.NamedTokenizer namedTokenizer) {
        String encoding = namedTokenizer.name().replace("jtokkit-", "");
        return TiktokenFixtures.encoding(encoding);
    }

    private static Stream<TiktokenFixtures.NamedTokenizer> tokenizers() {
        return TiktokenFixtures.createAllJtokkitTokenizers().stream();
    }
}

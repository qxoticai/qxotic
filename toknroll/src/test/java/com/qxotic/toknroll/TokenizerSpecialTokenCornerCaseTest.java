package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerSpecialTokenCornerCaseTest {

    @ParameterizedTest(name = "special token rejected in encode {0}")
    @MethodSource("tokenizers")
    void specialTokenTextIsRejectedDuringEncode(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        String[] texts = {"<|endoftext|>", "Hello <|endoftext|> world", "Start <|endoftext|> end"};
        for (String text : texts) {
            assertThrows(
                    UnsupportedOperationException.class,
                    () -> tokenizer.encode(text),
                    namedTokenizer.name() + " should reject " + text);
        }
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

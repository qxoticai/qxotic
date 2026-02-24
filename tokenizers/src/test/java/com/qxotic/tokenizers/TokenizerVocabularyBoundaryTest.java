package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.testkit.TiktokenFixtures;
import java.util.NoSuchElementException;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerVocabularyBoundaryTest {

    @ParameterizedTest(name = "decode invalid ids {0}")
    @MethodSource("tokenizers")
    void decodeRejectsInvalidTokenIds(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        IntSequence[] invalid = {
            IntSequence.of(-1),
            IntSequence.of(Integer.MAX_VALUE),
            IntSequence.of(tokenizer.vocabulary().size() + 1000)
        };

        for (IntSequence sequence : invalid) {
            assertThrows(Exception.class, () -> tokenizer.decode(sequence), namedTokenizer.name());
            assertThrows(
                    Exception.class, () -> tokenizer.decodeBytes(sequence), namedTokenizer.name());
        }
    }

    @ParameterizedTest(name = "vocab excludes invalid ids {0}")
    @MethodSource("tokenizers")
    void vocabularyExcludesInvalidIds(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Vocabulary vocab = namedTokenizer.tokenizer().vocabulary();
        assertFalse(vocab.contains(-1), namedTokenizer.name() + " negative id");
        assertFalse(vocab.contains(Integer.MAX_VALUE), namedTokenizer.name() + " max int");
        assertFalse(vocab.contains(vocab.size()), namedTokenizer.name() + " id at size");
        assertTrue(vocab.size() > 50_000, namedTokenizer.name() + " expected large vocab");
    }

    @ParameterizedTest(name = "vocab lookup throws {0}")
    @MethodSource("tokenizers")
    void vocabularyLookupThrowsForInvalidIds(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Vocabulary vocab = namedTokenizer.tokenizer().vocabulary();
        int[] invalidIds = {-1, vocab.size(), Integer.MAX_VALUE};
        for (int id : invalidIds) {
            assertThrows(
                    NoSuchElementException.class, () -> vocab.token(id), namedTokenizer.name());
        }
    }

    private static Stream<TiktokenFixtures.NamedTokenizer> tokenizers() {
        return TiktokenFixtures.createAllJtokkitTokenizers().stream();
    }
}

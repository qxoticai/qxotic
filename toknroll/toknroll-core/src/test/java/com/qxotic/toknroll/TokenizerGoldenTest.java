package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenGoldenFixture;
import com.qxotic.toknroll.testkit.TiktokenGoldenFixture.CaseData;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerGoldenTest {

    private static final List<String> ENCODINGS = List.of("r50k_base", "cl100k_base", "o200k_base");

    /**
     * Null when not fetched: like the vocabularies it pairs with, the truth is fetched, not
     * tracked.
     */
    private static TiktokenGoldenFixture fixture() {
        try {
            return TiktokenGoldenFixture.load();
        } catch (IllegalStateException e) {
            return null;
        }
    }

    /** The one case that shows up as SKIPPED with the remedy when the truth was not fetched. */
    @Test
    void goldenTruthIsFetched() {
        Assumptions.assumeTrue(
                fixture() != null, "golden fixture not fetched - run `make test-fixtures`");
    }

    @ParameterizedTest(name = "golden {0}/{1}", allowZeroInvocations = true)
    @MethodSource("goldenCases")
    void goldenEncodingMatchesExpected(
            String encoding,
            String caseId,
            String text,
            int[] expectedTokens,
            String expectedDecoded,
            byte[] expectedDecodedBytes,
            int expectedTokenCount) {

        Tokenizer tokenizer = TestTokenizers.tiktokenReference(encoding);
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
        TiktokenGoldenFixture fixture = fixture();
        if (fixture == null) return Stream.empty(); // goldenTruthIsFetched says why
        List<Arguments> args = new ArrayList<>();
        for (String encoding : ENCODINGS) {
            for (CaseData c : fixture.getCases(encoding)) {
                args.add(
                        Arguments.of(
                                encoding,
                                c.caseId(),
                                c.inputText(),
                                c.tokens(),
                                c.decoded(),
                                c.decodedBytes(),
                                c.tokenCount()));
            }
        }
        return args.stream();
    }
}

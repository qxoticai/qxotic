package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.TiktokenGoldenFixture;
import com.qxotic.toknroll.testkit.TiktokenGoldenFixture.CaseData;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerGoldenTest {

    private static final TiktokenGoldenFixture FIXTURE = TiktokenGoldenFixture.load();
    private static final List<EncodingSpec> ENCODINGS =
            List.of(
                    new EncodingSpec("r50k_base", Integer.MAX_VALUE),
                    new EncodingSpec("cl100k_base", 25),
                    new EncodingSpec("o200k_base", 25));

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
        List<Arguments> args = new ArrayList<>();
        for (EncodingSpec spec : ENCODINGS) {
            int limit = Math.min(spec.maxCases(), FIXTURE.getCases(spec.name()).size());
            for (CaseData c : FIXTURE.getSampledCases(spec.name(), limit)) {
                args.add(
                        Arguments.of(
                                spec.name(),
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

    private record EncodingSpec(String name, int maxCases) {}
}

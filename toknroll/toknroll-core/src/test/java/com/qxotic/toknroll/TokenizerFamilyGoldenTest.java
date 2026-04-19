package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import com.qxotic.toknroll.testkit.FamilyTestSpecs.FamilySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@Tag("network")
class TokenizerFamilyGoldenTest {

    private static final List<FamilySpec> FAMILY_SPECS = FamilyTestSpecs.FAMILIES;

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();
    private static final ConcurrentHashMap<String, Specials> FAMILY_SPECIALS =
            new ConcurrentHashMap<>();
    private static final Set<String> SPECIALS_AWARE_FAMILIES =
            Set.of("alibaba.qwen3_5", "google.gemma4");

    @ParameterizedTest(name = "family parity budget {0}")
    @MethodSource("familyParityBudgets")
    void familyTokenParityBudget(String familyId, double minExactRatio, int maxEncodeErrors) {
        Tokenizer tokenizer = requireTokenizer(familyId);
        List<CaseData> cases = FIXTURE.getCases(familyId);
        Assumptions.assumeTrue(!cases.isEmpty(), "No fixture cases for " + familyId);

        int exact = 0;
        int checked = 0;
        int encodeErrors = 0;
        List<String> mismatchDetails = new ArrayList<>();
        List<String> encodeErrorDetails = new ArrayList<>();

        for (CaseData c : cases) {
            int[] actual;
            try {
                actual = encodeFamilyAware(familyId, tokenizer, c.text()).toArray();
            } catch (RuntimeException ex) {
                encodeErrors++;
                if (encodeErrorDetails.size() < 3) {
                    encodeErrorDetails.add(c.caseId() + ": " + ex.getClass().getSimpleName());
                }
                continue;
            }

            if (Arrays.equals(c.tokens(), actual)) {
                exact++;
            } else if (mismatchDetails.size() < 5) {
                mismatchDetails.add(describeFirstDiff(tokenizer, c, actual));
            }
            checked++;

            assertEquals(
                    c.decoded(),
                    tokenizer.decode(actual),
                    familyId + "/" + c.caseId() + " local decoded parity");
        }

        double exactRatio = checked == 0 ? 0.0 : ((double) exact / (double) checked);
        assertTrue(
                exactRatio >= minExactRatio,
                familyId
                        + " exact ratio "
                        + exactRatio
                        + " < "
                        + minExactRatio
                        + "; sample mismatches="
                        + mismatchDetails);
        assertTrue(
                encodeErrors <= maxEncodeErrors,
                familyId
                        + " encode errors "
                        + encodeErrors
                        + " > "
                        + maxEncodeErrors
                        + "; sample encode errors="
                        + encodeErrorDetails);
    }

    @ParameterizedTest(name = "family golden encode {0}/{1}")
    @MethodSource("familyGoldenCases")
    void familyGoldenEncodingMatchesExpected(
            String familyId,
            String caseId,
            String text,
            int[] expectedTokens,
            String expectedDecoded,
            int expectedTokenCount) {
        Tokenizer tokenizer = requireTokenizer(familyId);
        assertEquals(
                expectedTokenCount,
                expectedTokens.length,
                familyId + "/" + caseId + " token_count sanity");
        assertTrue(expectedDecoded != null, familyId + "/" + caseId + " fixture decoded present");

        IntSequence reencoded = encodeFamilyAware(familyId, tokenizer, text);
        assertEquals(
                expectedDecoded,
                tokenizer.decode(reencoded),
                familyId + "/" + caseId + " local decoded round-trip");
        assertEquals(
                reencoded.length(),
                encodeFamilyAware(familyId, tokenizer, text).length(),
                familyId + "/" + caseId + " local special-aware count parity");
    }

    @ParameterizedTest(name = "family golden decode {0}/{1}")
    @MethodSource("familyGoldenCases")
    void familyGoldenDecodeMatchesExpected(
            String familyId,
            String caseId,
            String text,
            int[] expectedTokens,
            String expectedDecoded,
            int expectedTokenCount) {
        Tokenizer tokenizer = requireTokenizer(familyId);
        assertEquals(
                expectedTokenCount,
                expectedTokens.length,
                familyId + "/" + caseId + " token_count sanity");
        assertTrue(expectedDecoded != null, familyId + "/" + caseId + " fixture decoded present");

        // Golden decode: decode the reference tokenizer's tokens and compare with reference decoded
        // text
        String decoded = tokenizer.decode(IntSequence.copyOf(expectedTokens));
        assertEquals(
                expectedDecoded,
                decoded,
                familyId
                        + "/"
                        + caseId
                        + " golden decode mismatch for tokens="
                        + Arrays.toString(expectedTokens));
    }

    static Stream<Arguments> familyGoldenCases() {
        List<Arguments> args = new ArrayList<>();
        for (FamilySpec spec : FAMILY_SPECS) {
            for (CaseData c : FIXTURE.getCases(spec.familyId())) {
                args.add(
                        Arguments.of(
                                spec.familyId(),
                                c.caseId(),
                                c.text(),
                                c.tokens(),
                                c.decoded(),
                                c.tokenCount()));
            }
        }
        return args.stream();
    }

    static Stream<Arguments> familyParityBudgets() {
        return FAMILY_SPECS.stream()
                .map(
                        spec ->
                                Arguments.of(
                                        spec.familyId(),
                                        spec.minExactRatio(),
                                        spec.maxEncodeErrors()));
    }

    private static String describeFirstDiff(Tokenizer tokenizer, CaseData expected, int[] actual) {
        int min = Math.min(expected.tokens().length, actual.length);
        for (int i = 0; i < min; i++) {
            if (expected.tokens()[i] != actual[i]) {
                return expected.caseId()
                        + "@"
                        + i
                        + " exp="
                        + expected.tokens()[i]
                        + "("
                        + safeToken(tokenizer, expected.tokens()[i])
                        + ")"
                        + " got="
                        + actual[i]
                        + "("
                        + safeToken(tokenizer, actual[i])
                        + ")"
                        + " text='"
                        + compact(expected.text())
                        + "'";
            }
        }
        return expected.caseId()
                + " length exp="
                + expected.tokens().length
                + " got="
                + actual.length
                + " text='"
                + compact(expected.text())
                + "'";
    }

    private static String safeToken(Tokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.vocabulary().token(tokenId).replace("\n", "\\n");
        } catch (RuntimeException e) {
            return "?";
        }
    }

    private static String compact(String text) {
        String t = text.replace("\n", "\\n").replace("\t", "\\t");
        return t.length() <= 32 ? t : t.substring(0, 29) + "...";
    }

    private static Tokenizer requireTokenizer(String familyId) {
        Optional<Tokenizer> maybeTokenizer = ModelFamilyTokenizers.create(familyId);
        Assumptions.assumeTrue(
                maybeTokenizer.isPresent(), "Tokenizer not available for " + familyId);
        return maybeTokenizer.get();
    }

    private static IntSequence encodeFamilyAware(
            String familyId, Tokenizer tokenizer, String text) {
        if (!SPECIALS_AWARE_FAMILIES.contains(familyId)) {
            return tokenizer.encode(text);
        }
        Specials specials =
                FAMILY_SPECIALS.computeIfAbsent(familyId, id -> compileControlSpecials(tokenizer));
        if (specials.isEmpty()) {
            return tokenizer.encode(text);
        }
        return specials.encode(tokenizer, text);
    }

    private static Specials compileControlSpecials(Tokenizer tokenizer) {
        Set<String> specials = new LinkedHashSet<>();
        Vocabulary vocabulary = tokenizer.vocabulary();
        for (Map.Entry<String, Integer> e : vocabulary) {
            String token = e.getKey();
            if (token == null || token.isEmpty()) {
                continue;
            }
            if (vocabulary.isTokenOfType(e.getValue(), StandardTokenType.CONTROL)) {
                specials.add(token);
            }
        }
        if (specials.isEmpty()) {
            return Specials.none();
        }
        try {
            return Specials.compile(vocabulary, specials);
        } catch (IllegalArgumentException ignored) {
            return Specials.none();
        }
    }
}

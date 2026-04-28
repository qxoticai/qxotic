package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import com.qxotic.toknroll.testkit.FamilyTestSpecs.FamilySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@Tag("slow")
@Tag("local-external")
class HuggingFaceFamilyGoldenTest {

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();
    private static final ConcurrentHashMap<String, Tokenizer> TOKENIZERS =
            new ConcurrentHashMap<>();

    @ParameterizedTest(name = "hf family token parity budget {0}")
    @MethodSource("familyParityBudgets")
    void familyTokenParityBudget(
            String familyId,
            String modelRef,
            String revision,
            double minExactRatio,
            int maxEncodeErrors) {
        Tokenizer tokenizer = requireTokenizer(familyId, modelRef, revision);
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
                actual = tokenizer.encode(c.text()).toArray();
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
                    tokenizer.decode(IntSequence.of(actual)),
                    familyId + "/" + c.caseId() + " decoded parity");
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

    static Stream<Arguments> familyParityBudgets() {
        List<Arguments> args = new ArrayList<>();
        for (FamilySpec spec : FamilyTestSpecs.FAMILIES) {
            Family family = FIXTURE.families().get(spec.familyId());
            if (family == null || family.modelRef() == null || family.modelRef().isBlank()) {
                continue;
            }
            args.add(
                    Arguments.of(
                            spec.familyId(),
                            family.modelRef(),
                            family.revision() == null || family.revision().isBlank()
                                    ? "main"
                                    : family.revision(),
                            spec.minExactRatio(),
                            spec.maxEncodeErrors()));
        }
        return args.stream();
    }

    private static Tokenizer requireTokenizer(String familyId, String modelRef, String revision) {
        return TOKENIZERS.computeIfAbsent(
                familyId,
                ignored -> {
                    String[] parts = modelRef.split("/", 2);
                    Assumptions.assumeTrue(
                            parts.length == 2,
                            "Invalid model_ref for " + familyId + ": " + modelRef);
                    return HuggingFaceTokenizerLoader.fromHuggingFace(
                            parts[0], parts[1], revision, false, false);
                });
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
}

package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import com.qxotic.toknroll.testkit.FamilyTestSpecs.FamilySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    private static final ConcurrentHashMap<String, Specials> FAMILY_SPECIALS =
            new ConcurrentHashMap<>();
    private static final Set<String> BRACKET_SPECIAL_FAMILIES = Set.of("mistral.gpt2_pretekken");
    private static final Map<String, String> MODEL_REF_OVERRIDES =
            Map.of(
                    "google/gemma-3-4b-it", "unsloth/gemma-3-4b-it",
                    "meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct");

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
                    String effectiveModelRef = MODEL_REF_OVERRIDES.getOrDefault(modelRef, modelRef);
                    String[] parts = effectiveModelRef.split("/", 2);
                    String effectiveRevision =
                            effectiveModelRef.equals(modelRef) ? revision : "main";
                    Assumptions.assumeTrue(
                            parts.length == 2,
                            "Invalid model_ref for " + familyId + ": " + effectiveModelRef);
                    return HuggingFaceTokenizerLoader.fromHuggingFace(
                            parts[0], parts[1], effectiveRevision, false, false);
                });
    }

    private static IntSequence encodeFamilyAware(
            String familyId, Tokenizer tokenizer, String text) {
        Specials specials =
                FAMILY_SPECIALS.computeIfAbsent(
                        familyId, ignored -> compileControlSpecials(familyId, tokenizer));
        if (specials.isEmpty()) {
            return tokenizer.encode(text);
        }
        return specials.encode(tokenizer, text);
    }

    private static Specials compileControlSpecials(String familyId, Tokenizer tokenizer) {
        Set<String> specials = new LinkedHashSet<>();
        Vocabulary vocabulary = tokenizer.vocabulary();
        for (Map.Entry<String, Integer> e : vocabulary) {
            String token = e.getKey();
            if (token == null || token.isEmpty()) {
                continue;
            }
            if (vocabulary.isTokenOfType(e.getValue(), StandardTokenType.CONTROL)
                    || looksLikeSpecialToken(familyId, token)) {
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

    private static boolean looksLikeSpecialToken(String familyId, String token) {
        if (token.length() < 3) {
            return false;
        }
        if (token.startsWith("<|") && token.endsWith("|>")) {
            return true;
        }
        if (BRACKET_SPECIAL_FAMILIES.contains(familyId)
                && token.startsWith("[")
                && token.endsWith("]")) {
            return token.equals(token.toUpperCase());
        }
        return false;
    }
}

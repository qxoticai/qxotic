package com.qxotic.toknroll.hf;

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
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class HuggingFaceTokenizerSourceComparisonTest {

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();
    private static final boolean VERBOSE =
            Boolean.getBoolean("toknroll.hf.sourceComparison.verbose");
    private static final ConcurrentHashMap<String, Specials> FAMILY_SPECIALS =
            new ConcurrentHashMap<>();
    private static final Set<String> BRACKET_SPECIAL_FAMILIES = Set.of("mistral.gpt2_pretekken");
    private static final Map<String, String> MODEL_REF_OVERRIDES =
            Map.of(
                    "google/gemma-3-4b-it", "unsloth/gemma-3-4b-it",
                    "meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct");

    @Test
    void comparePinnedRevisionVsMainAgainstGoldenFixture() {
        int compared = 0;
        for (FamilyTestSpecs.FamilySpec spec : FamilyTestSpecs.FAMILIES) {
            String familyId = spec.familyId();
            Family family = FIXTURE.families().get(familyId);
            if (family == null || family.modelRef() == null || family.modelRef().isBlank()) {
                continue;
            }

            String effectiveModelRef =
                    MODEL_REF_OVERRIDES.getOrDefault(family.modelRef(), family.modelRef());
            String[] refParts = effectiveModelRef.split("/", 2);
            if (refParts.length != 2) {
                continue;
            }

            String user = refParts[0];
            String repository = refParts[1];
            String pinnedRevision =
                    family.revision() == null || family.revision().isBlank()
                            ? "main"
                            : family.revision();
            if (!effectiveModelRef.equals(family.modelRef())) {
                pinnedRevision = "main";
            }

            Tokenizer pinned =
                    HuggingFaceTokenizerLoader.fromHuggingFace(
                            user, repository, pinnedRevision, false, false);
            Tokenizer main =
                    HuggingFaceTokenizerLoader.fromHuggingFace(
                            user, repository, "main", false, false);

            List<CaseData> cases = FIXTURE.getCases(familyId);
            if (cases.isEmpty()) {
                continue;
            }

            TokenizerSourceStats pinnedStats = evaluate(familyId, pinned, cases);
            TokenizerSourceStats mainStats = evaluate(familyId, main, cases);
            if (VERBOSE) {
                System.out.println(
                        "[hf-source-compare] "
                                + familyId
                                + " pinned="
                                + pinnedStats
                                + " main="
                                + mainStats
                                + " model_ref="
                                + effectiveModelRef
                                + " pinned_revision="
                                + pinnedRevision);
            }
            compared++;
        }

        assertTrue(compared > 0, "No family tokenizers available for source comparison");
    }

    private static TokenizerSourceStats evaluate(
            String familyId, Tokenizer tokenizer, List<CaseData> cases) {
        int exact = 0;
        int checked = 0;
        int encodeErrors = 0;
        for (CaseData c : cases) {
            try {
                int[] actual = encodeFamilyAware(familyId, tokenizer, c.text()).toArray();
                if (Arrays.equals(c.tokens(), actual)) {
                    exact++;
                }
                checked++;
            } catch (RuntimeException ex) {
                encodeErrors++;
            }
        }
        double ratio = checked == 0 ? 0d : (double) exact / (double) checked;
        return new TokenizerSourceStats(exact, checked, encodeErrors, ratio);
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

    private static final class TokenizerSourceStats {
        private final int exact;
        private final int checked;
        private final int encodeErrors;
        private final double ratio;

        private TokenizerSourceStats(int exact, int checked, int encodeErrors, double ratio) {
            this.exact = exact;
            this.checked = checked;
            this.encodeErrors = encodeErrors;
            this.ratio = ratio;
        }

        @Override
        public String toString() {
            return "{exact="
                    + exact
                    + ", checked="
                    + checked
                    + ", encodeErrors="
                    + encodeErrors
                    + ", ratio="
                    + ratio
                    + "}";
        }
    }
}

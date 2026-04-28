package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class HuggingFaceTokenizerSourceComparisonTest {

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();
    private static final boolean VERBOSE =
            Boolean.getBoolean("toknroll.hf.sourceComparison.verbose");

    @Test
    void comparePinnedRevisionVsMainAgainstGoldenFixture() {
        int compared = 0;
        for (FamilyTestSpecs.FamilySpec spec : FamilyTestSpecs.FAMILIES) {
            String familyId = spec.familyId();
            Family family = FIXTURE.families().get(familyId);
            if (family == null || family.modelRef() == null || family.modelRef().isBlank()) {
                continue;
            }

            String[] refParts = family.modelRef().split("/", 2);
            if (refParts.length != 2) {
                continue;
            }

            String user = refParts[0];
            String repository = refParts[1];
            String pinnedRevision =
                    family.revision() == null || family.revision().isBlank()
                            ? "main"
                            : family.revision();

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
                                + family.modelRef()
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
                int[] actual = tokenizer.encode(c.text()).toArray();
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

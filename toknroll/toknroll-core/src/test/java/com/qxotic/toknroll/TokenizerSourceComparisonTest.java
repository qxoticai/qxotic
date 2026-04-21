package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
class TokenizerSourceComparisonTest {

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();

    @Test
    void compareGgufVsHfTokenizerSources() {
        int compared = 0;
        for (FamilyTestSpecs.FamilySpec spec : FamilyTestSpecs.FAMILIES) {
            String familyId = spec.familyId();
            Family family = FIXTURE.families().get(familyId);
            if (family == null || family.modelRef() == null) {
                continue;
            }

            Optional<Tokenizer> gguf = ModelFamilyTokenizers.create(familyId);
            Optional<Tokenizer> hf =
                    ModelFamilyTokenizers.createFromHfFiles(
                            familyId, family.modelRef(), family.revision());
            if (gguf.isEmpty() || hf.isEmpty()) {
                continue;
            }

            TokenizerSourceStats ggufStats = evaluate(gguf.get(), FIXTURE.getCases(familyId));
            TokenizerSourceStats hfStats = evaluate(hf.get(), FIXTURE.getCases(familyId));

            System.out.println(
                    "[source-compare] "
                            + familyId
                            + " gguf="
                            + ggufStats
                            + " hf="
                            + hfStats
                            + " model_ref="
                            + family.modelRef());
            compared++;
        }
        assertTrue(compared > 0, "No family tokenizers available for source comparison");
    }

    private static TokenizerSourceStats evaluate(Tokenizer tokenizer, List<CaseData> cases) {
        int exact = 0;
        int checked = 0;
        int encodeErrors = 0;
        for (CaseData c : cases) {
            try {
                int[] actual = tokenizer.encodeToArray(c.text());
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
}

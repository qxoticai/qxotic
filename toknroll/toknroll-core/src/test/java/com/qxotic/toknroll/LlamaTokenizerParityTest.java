package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
class LlamaTokenizerParityTest {

    private static final String FAMILY_ID = "meta.llama3";
    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();

    @Test
    void ggufLlama3MatchesGoldenFixture() {
        Optional<Tokenizer> gguf = ModelFamilyTokenizers.create(FAMILY_ID);
        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);

        List<CaseData> cases = sampledCases();
        int exact = 0;
        for (CaseData c : cases) {
            int[] actual = gguf.get().encodeToArray(c.text());
            if (java.util.Arrays.equals(c.tokens(), actual)) {
                exact++;
            }
            assertEquals(
                    c.tokens().length,
                    gguf.get().countTokens(c.text()),
                    "countTokens mismatch for case " + c.caseId());
        }

        double exactRatio = cases.isEmpty() ? 0d : (double) exact / (double) cases.size();
        assertEquals(1.0d, exactRatio, 0.0d, "GGUF llama3 exact ratio must be 100%");
    }

    @Test
    void ggufAndHfLlama3AgreeOnGoldenCases() {
        Family family = FIXTURE.families().get(FAMILY_ID);
        assumeTrue(family != null, "Missing fixture family " + FAMILY_ID);

        Optional<Tokenizer> gguf = ModelFamilyTokenizers.create(FAMILY_ID);
        Optional<Tokenizer> hf =
                ModelFamilyTokenizers.createFromHfFiles(
                        FAMILY_ID, family.modelRef(), family.revision());

        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);
        assumeTrue(hf.isPresent(), "HF tokenizer unavailable for " + FAMILY_ID);

        List<CaseData> cases = sampledCases();
        assertTrue(!cases.isEmpty(), "No sampled cases for " + FAMILY_ID);

        for (CaseData c : cases) {
            int[] ggufTokens = gguf.get().encodeToArray(c.text());
            int[] hfTokens = hf.get().encodeToArray(c.text());
            assertArrayEquals(hfTokens, ggufTokens, "HF!=GGUF for case " + c.caseId());
        }
    }

    private static List<CaseData> sampledCases() {
        List<CaseData> cases = FIXTURE.getCases(FAMILY_ID);
        assertTrue(!cases.isEmpty(), "No golden cases for " + FAMILY_ID);
        return cases;
    }
}

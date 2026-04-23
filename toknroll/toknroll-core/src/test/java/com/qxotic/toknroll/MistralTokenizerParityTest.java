package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.TestTokenizers;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
class MistralTokenizerParityTest {

    private static final String FAMILY_ID = "mistral.v0_3";
    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();

    @Test
    void ggufMistralMatchesGoldenFixture() {
        Optional<Tokenizer> gguf = TestTokenizers.modelFamily(FAMILY_ID);
        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);

        List<FamilyGoldenFixture.CaseData> cases = sampledCases();
        int exact = 0;
        for (FamilyGoldenFixture.CaseData c : cases) {
            int[] actual = gguf.get().encodeToArray(c.text());
            if (c.caseId().equals("case_001") || c.caseId().equals("case_002")) {
                System.out.println(
                        c.caseId()
                                + ": text='"
                                + c.text()
                                + "' golden="
                                + Arrays.toString(c.tokens())
                                + " actual="
                                + Arrays.toString(actual));
            }
            if (Arrays.equals(c.tokens(), actual)) {
                exact++;
            }
            assertEquals(
                    c.tokens().length,
                    gguf.get().countTokens(c.text()),
                    "countTokens mismatch for case " + c.caseId());
        }

        double exactRatio = cases.isEmpty() ? 0d : (double) exact / (double) cases.size();
        assertEquals(1.0d, exactRatio, 0.0d, "GGUF mistral.v0_3 exact ratio must be 100%");
    }

    @Test
    void ggufAndHfMistralAgreeOnGoldenCases() {
        FamilyGoldenFixture.Family family = FIXTURE.families().get(FAMILY_ID);
        assumeTrue(family != null, "Missing fixture family " + FAMILY_ID);

        Optional<Tokenizer> gguf = TestTokenizers.modelFamily(FAMILY_ID);
        Optional<Tokenizer> hf =
                TestTokenizers.modelFamilyFromHf(FAMILY_ID, family.modelRef(), family.revision());

        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);
        assumeTrue(hf.isPresent(), "HF tokenizer unavailable for " + FAMILY_ID);

        List<FamilyGoldenFixture.CaseData> cases = sampledCases();
        assertTrue(!cases.isEmpty(), "No sampled cases for " + FAMILY_ID);

        for (FamilyGoldenFixture.CaseData c : cases) {
            int[] ggufTokens = gguf.get().encodeToArray(c.text());
            int[] hfTokens = hf.get().encodeToArray(c.text());
            assertArrayEquals(hfTokens, ggufTokens, "HF!=GGUF for case " + c.caseId());
        }
    }

    @Test
    void ggufMistralDecodeMatchesGolden() {
        Optional<Tokenizer> gguf = TestTokenizers.modelFamily(FAMILY_ID);
        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);

        Tokenizer tokenizer = gguf.get();
        List<FamilyGoldenFixture.CaseData> cases = sampledCases();

        for (FamilyGoldenFixture.CaseData c : cases) {
            // Decode golden reference tokens should match golden decoded text
            String decoded = tokenizer.decode(IntSequence.copyOf(c.tokens()));
            assertEquals(c.decoded(), decoded, "Decode mismatch for golden tokens " + c.caseId());

            // Round-trip: encode then decode should produce the golden decoded text
            int[] actual = tokenizer.encodeToArray(c.text());
            String roundTrip = tokenizer.decode(IntSequence.of(actual));
            assertEquals(
                    c.decoded(), roundTrip, "Round-trip decode mismatch for case " + c.caseId());

            // countBytes sanity check
            assertEquals(
                    c.tokenCount(), actual.length, "Token count mismatch for case " + c.caseId());
        }
    }

    private static List<FamilyGoldenFixture.CaseData> sampledCases() {
        List<FamilyGoldenFixture.CaseData> cases = FIXTURE.getCases(FAMILY_ID);
        assertTrue(!cases.isEmpty(), "No golden cases for " + FAMILY_ID);
        return cases;
    }
}

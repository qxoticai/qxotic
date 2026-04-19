package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
class Gemma4SentencePieceBpeGoldenTest {

    private static final String FAMILY_ID = "google.gemma4";
    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();
    private static volatile Specials specials;

    @Test
    void ggufGemma4SentencePieceBpeMatchesGoldenFixture() {
        Optional<Tokenizer> gguf = ModelFamilyTokenizers.create(FAMILY_ID);
        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);

        Tokenizer tokenizer = gguf.get();
        Specials familySpecials = specialsFor(tokenizer);

        List<CaseData> cases = sampledCases();
        assumeTrue(!cases.isEmpty(), "No sampled golden cases for " + FAMILY_ID);

        for (CaseData c : cases) {
            int[] actual = familySpecials.encode(tokenizer, c.text()).toArray();
            assertArrayEquals(c.tokens(), actual, "Token mismatch for case " + c.caseId());
            assertEquals(
                    c.decoded(),
                    tokenizer.decode(IntSequence.of(actual)),
                    "Decoded mismatch for case " + c.caseId());
            assertEquals(
                    c.tokenCount(),
                    actual.length,
                    "special-aware token_count mismatch for case " + c.caseId());
        }
    }

    @Test
    void ggufAndHfGemma4AgreeOnGoldenCases() {
        Family family = FIXTURE.families().get(FAMILY_ID);
        assumeTrue(family != null, "Missing fixture family " + FAMILY_ID);

        Optional<Tokenizer> gguf = ModelFamilyTokenizers.create(FAMILY_ID);
        Optional<Tokenizer> hf =
                ModelFamilyTokenizers.createFromHfFiles(
                        FAMILY_ID, family.modelRef(), family.revision());

        assumeTrue(gguf.isPresent(), "GGUF tokenizer unavailable for " + FAMILY_ID);
        assumeTrue(hf.isPresent(), "HF tokenizer unavailable for " + FAMILY_ID);
        Specials ggufSpecials = specialsFor(gguf.get());
        Specials hfSpecials = specialsFor(hf.get());

        List<CaseData> cases = sampledCases();
        assumeTrue(!cases.isEmpty(), "No sampled golden cases for " + FAMILY_ID);

        for (CaseData c : cases) {
            int[] ggufTokens = ggufSpecials.encode(gguf.get(), c.text()).toArray();
            int[] hfTokens = hfSpecials.encode(hf.get(), c.text()).toArray();
            assertArrayEquals(hfTokens, ggufTokens, "HF!=GGUF for case " + c.caseId());
        }
    }

    private static List<CaseData> sampledCases() {
        return FIXTURE.getCases(FAMILY_ID);
    }

    private static Specials specialsFor(Tokenizer tokenizer) {
        Specials local = specials;
        if (local != null) {
            return local;
        }
        synchronized (Gemma4SentencePieceBpeGoldenTest.class) {
            if (specials != null) {
                return specials;
            }
            Set<String> control = new LinkedHashSet<>();
            Vocabulary vocabulary = tokenizer.vocabulary();
            for (Map.Entry<String, Integer> e : vocabulary) {
                String token = e.getKey();
                if (token == null || token.isEmpty()) {
                    continue;
                }
                if (vocabulary.isTokenOfType(e.getValue(), StandardTokenType.CONTROL)) {
                    control.add(token);
                }
            }
            specials = control.isEmpty() ? Specials.none() : Specials.compile(vocabulary, control);
            return specials;
        }
    }
}

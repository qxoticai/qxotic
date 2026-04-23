package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import com.qxotic.toknroll.testkit.TestTokenizers;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Smoke contract tests for family tokenizer recreation.
 *
 * <p>These tests intentionally delegate tokenizer creation to {@link TestTokenizers} so all
 * recreation logic has a single source of truth.
 */
@Tag("network")
class ModelFamilyTokenizerRecreationTest {

    private static final List<String> SMOKE_TEXTS = FamilyTestSpecs.SMOKE_TEXTS;

    @ParameterizedTest(name = "recreate tokenizer family {0}")
    @MethodSource("families")
    void recreateTokenizerFromModelFamilyMetadata(String familyId) {
        Optional<Tokenizer> maybeTokenizer = TestTokenizers.modelFamily(familyId);
        Assumptions.assumeTrue(maybeTokenizer.isPresent(), familyId + " tokenizer not available");

        Tokenizer tokenizer = maybeTokenizer.get();
        assertNotNull(tokenizer, familyId + " tokenizer");
        assertTrue(tokenizer.vocabulary().size() > 0, familyId + " non-empty vocabulary");

        for (String text : SMOKE_TEXTS) {
            IntSequence tokens = tokenizer.encode(text);
            assertEquals(tokens.length(), tokenizer.countTokens(text), familyId + " count parity");
            assertFalse(tokens.isEmpty(), familyId + " non-empty tokens");
            for (int i = 0; i < tokens.length(); i++) {
                assertTrue(
                        tokenizer.vocabulary().contains(tokens.intAt(i)),
                        familyId + " token exists in vocabulary @" + i);
            }
            assertNotNull(tokenizer.decode(tokens), familyId + " decoded string");
        }
    }

    private static Stream<Arguments> families() {
        return FamilyTestSpecs.FAMILIES.stream().map(spec -> Arguments.of(spec.familyId()));
    }
}

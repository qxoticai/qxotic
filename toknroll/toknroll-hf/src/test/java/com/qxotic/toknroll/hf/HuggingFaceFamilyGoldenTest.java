package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.CaseData;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import com.qxotic.toknroll.testkit.FamilyTestSpecs;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Hugging Face's ids and text for each family's golden cases, from {@code
 * toknroll-benchmarks/generate_ground_truth.py}, with the tokenizer.json at the pinned revision and
 * at main. Every divergence fails, all of them listed, a known one with its cause. At main, one
 * also fails when the repository moved past the pinned revision: the golden is then due for
 * regeneration.
 */
@Tag("local-external")
class HuggingFaceFamilyGoldenTest {

    private static final FamilyGoldenFixture FIXTURE = FamilyGoldenFixture.load();

    @ParameterizedTest(name = "{0}@{1}")
    @MethodSource("families")
    void everyCaseMatches(String familyId, String revision) {
        Family family = FIXTURE.families().get(familyId);
        assertNotNull(family, familyId + " is missing from the golden");
        assertFalse(family.cases().isEmpty(), familyId + " has no golden cases");
        String[] ownerRepo = family.modelRef().split("/", 2);
        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.fromHuggingFace(
                        ownerRepo[0], ownerRepo[1], revision, false, false);
        List<String> divergences = new ArrayList<>();
        for (CaseData c : family.cases()) {
            int[] ids = tokenizer.encode(c.text()).toArray();
            if (!Arrays.equals(c.tokens(), ids)) {
                divergences.add(
                        firstDiff(tokenizer, c, ids)
                                + (c.known() == null ? "" : " known divergence: " + c.known()));
            } else if (!c.decoded().equals(tokenizer.decode(IntSequence.wrap(ids)))) {
                divergences.add(c.caseId() + " decode");
            }
        }
        assertTrue(
                divergences.isEmpty(),
                familyId
                        + "@"
                        + revision
                        + ": "
                        + divergences.size()
                        + " of "
                        + family.cases().size()
                        + " cases diverge: "
                        + divergences);
    }

    static Stream<Arguments> families() {
        return FamilyTestSpecs.FAMILIES.stream()
                .flatMap(
                        id -> {
                            Family family = FIXTURE.families().get(id);
                            String pinned =
                                    family == null
                                                    || family.revision() == null
                                                    || family.revision().isBlank()
                                            ? "main"
                                            : family.revision();
                            return pinned.equals("main")
                                    ? Stream.of(Arguments.of(id, "main"))
                                    : Stream.of(Arguments.of(id, pinned), Arguments.of(id, "main"));
                        });
    }

    private static String firstDiff(Tokenizer tokenizer, CaseData expected, int[] actual) {
        int[] tokens = expected.tokens();
        int i = Arrays.mismatch(tokens, actual);
        String text = expected.text().replace("\n", "\\n").replace("\t", "\\t");
        String where =
                i < Math.min(tokens.length, actual.length)
                        ? "@"
                                + i
                                + " expected "
                                + shown(tokenizer, tokens[i])
                                + " got "
                                + shown(tokenizer, actual[i])
                        : " length expected " + tokens.length + " got " + actual.length;
        return expected.caseId()
                + where
                + " in '"
                + (text.length() <= 32 ? text : text.substring(0, 29) + "...")
                + "'";
    }

    private static String shown(Tokenizer tokenizer, int id) {
        return id + "(" + tokenizer.vocabulary().token(id).replace("\n", "\\n") + ")";
    }
}

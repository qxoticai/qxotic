package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.EdgeParityGolden;
import com.qxotic.toknroll.testkit.EdgeParityGolden.Expected;
import com.qxotic.toknroll.testkit.EdgeParityGolden.Model;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicContainer;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestFactory;

/**
 * Hugging Face's own answers on edge-case texts (leading and trailing spaces, empty text, CRLF,
 * byte fallback, emoji, special-token spellings, literal U+2581, long runs), across tokenizer.json
 * families, in both modes: plain encode against the tokenizer.json without its added tokens (raw
 * text: ids, text, bytes and counts), Specials.encode against it as it is (every added token
 * matched). The goldens come from {@code scripts/hf_edge_parity.py}, each model at its pinned
 * revision. Every divergence fails.
 *
 * <p>Tokenizers load from the hub at the golden's revision, or with {@code
 * -Dtoknroll.test.tokenizers=DIR} from {@code DIR/owner__repo/tokenizer.json}, which must be that
 * revision's file.
 */
@Tag("local-external")
@Tag("network")
class HuggingFaceEdgeParityTest {

    private static final String LOCAL = System.getProperty("toknroll.test.tokenizers");

    @TestFactory
    Stream<DynamicContainer> hfParity() {
        EdgeParityGolden golden =
                EdgeParityGolden.load(
                        HuggingFaceEdgeParityTest.class,
                        "hf_edge_parity.json",
                        "toknroll/scripts/hf_edge_parity.py");
        return golden.models().stream().map(model -> container(model, golden.texts()));
    }

    /** One model's cases; a tokenizer that does not load is one failing test with the error. */
    private static DynamicContainer container(Model model, List<String> texts) {
        String name = model.name() + "@" + model.revision();
        Tokenizer tokenizer;
        Specials specials;
        try {
            tokenizer = load(model);
            specials = Specials.of(tokenizer);
        } catch (RuntimeException e) {
            return dynamicContainer(
                    name,
                    Stream.of(
                            dynamicTest(
                                    "loads",
                                    () -> {
                                        throw e;
                                    })));
        }
        List<DynamicTest> tests = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            Expected expected = model.cases().get(i);
            String shown = EdgeParityGolden.printable(text);
            tests.add(dynamicTest(shown, () -> check(tokenizer, text, expected)));
            tests.add(
                    dynamicTest(
                            shown + " with specials",
                            () ->
                                    assertArrayEquals(
                                            expected.specials(),
                                            specials.encode(tokenizer, text).toArray(),
                                            "Specials.encode")));
        }
        return dynamicContainer(name, tests);
    }

    private static void check(Tokenizer tokenizer, String text, Expected expected) {
        int[] ids = expected.ids();
        assertArrayEquals(ids, tokenizer.encode(text).toArray(), "encode");
        assertEquals(expected.decoded(), tokenizer.decode(IntSequence.wrap(ids)), "decode");
        EdgeParityGolden.checkCountsAndBytes(
                text,
                ids,
                tokenizer::countTokens,
                t -> tokenizer.decode(IntSequence.wrap(t)),
                t -> tokenizer.decodeBytes(IntSequence.wrap(t)),
                t -> tokenizer.countBytes(IntSequence.wrap(t)));
    }

    private static Tokenizer load(Model model) {
        if (model.revision() == null) {
            throw new IllegalStateException(model.name() + " has no revision in the golden");
        }
        if (LOCAL != null) {
            Path file = Path.of(LOCAL, model.name().replace("/", "__"), "tokenizer.json");
            if (Files.exists(file)) {
                return HuggingFaceTokenizerLoader.fromLocal(file);
            }
        }
        String[] ownerRepo = model.name().split("/", 2);
        return HuggingFaceTokenizerLoader.fromHuggingFace(
                ownerRepo[0], ownerRepo[1], model.revision(), false, false);
    }
}

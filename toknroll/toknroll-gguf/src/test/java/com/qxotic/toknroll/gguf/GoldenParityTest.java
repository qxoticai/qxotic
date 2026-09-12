package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Byte-exact parity of a GGUF-loaded tokenizer against the model author's reference: every case in
 * the family's golden was encoded by Hugging Face {@code tokenizers} from the model's own {@code
 * tokenizer.json} ({@code add_special_tokens=False}, the same contract as {@link
 * Tokenizer#encode}). The tokenizer under test comes from the GGUF's metadata - the family's
 * pretokenizer registration, its vocabulary and merges - so the test pins the registration, not
 * just the BPE core. The golden is generated, not committed: {@code
 * toknroll-benchmarks/generate_laguna_golden.py --model ... --out ...} writes it, and the family
 * skips until it exists.
 *
 * <p>Out of scope by contract: special-token spellings inside the text. {@link Tokenizer#encode} is
 * the non-special-aware path (llama.cpp's default), so it encodes "〈|EOS|〉" as text, where the
 * reference's default would match the added token.
 */
@Tag("network")
@Tag("local-external")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class GoldenParityTest {

    /** One family: the GGUF whose metadata builds the tokenizer, and the golden it must match. */
    record Family(String cacheKey, String ggufUrl, String pre, String modelRef, String golden) {}

    private final Family family;
    private Tokenizer tokenizer;
    private Map<String, Object> golden;

    GoldenParityTest(Family family) {
        this.family = family;
    }

    @BeforeAll
    void load() {
        TestDataManager dataManager = new TestDataManager();
        GGUF gguf;
        try {
            gguf = dataManager.getOrDownloadMetadata(family.cacheKey(), family.ggufUrl());
        } catch (Exception e) {
            Assumptions.abort(family.modelRef() + " GGUF metadata unavailable: " + e);
            return;
        }
        assertEquals(family.pre(), gguf.getValueOrDefault(String.class, "tokenizer.ggml.pre", "?"));
        Path partial =
                dataManager
                        .getCachePath()
                        .resolve(TestDataManager.cacheFileNameForUrl(family.ggufUrl()));
        try {
            Path local = Files.createTempFile("toknroll-golden-", ".gguf");
            Files.copy(partial, local, StandardCopyOption.REPLACE_EXISTING);
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromLocal(local);
            Files.deleteIfExists(local);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        golden = readGolden();
    }

    @Test
    void goldenNamesTheReferenceItWasMadeFrom() {
        assertEquals(family.modelRef(), golden.get("model_ref"));
        assertEquals("add_special_tokens=False", golden.get("encode"));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void encodeMatchesTheReference(String id, String text, int[] expected) {
        assertArrayEquals(expected, tokenizer.encode(text).toArray(), id + ": token ids");
        assertEquals(expected.length, tokenizer.countTokens(text), id + ": countTokens");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void decodeRoundTripsTheReferenceIds(String id, String text, int[] expected) {
        assertEquals(text, tokenizer.decode(IntSequence.of(expected)), id + ": decode");
    }

    Stream<Arguments> cases() {
        if (golden == null) return Stream.empty(); // load() aborted: nothing to run
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> cases = (List<Map<String, Object>>) golden.get("cases");
        return cases.stream()
                .map(
                        c -> {
                            @SuppressWarnings("unchecked")
                            List<Number> ids = (List<Number>) c.get("tokens");
                            int[] tokens = ids.stream().mapToInt(Number::intValue).toArray();
                            return Arguments.of(c.get("id"), c.get("text"), tokens);
                        });
    }

    private Map<String, Object> readGolden() {
        try (InputStream in = GoldenParityTest.class.getResourceAsStream("/" + family.golden())) {
            if (in == null) {
                Assumptions.abort(
                        family.golden()
                                + " is not generated; run"
                                + " toknroll-benchmarks/generate_laguna_golden.py --model "
                                + family.modelRef()
                                + " --out toknroll-gguf/src/test/resources/"
                                + family.golden());
            }
            return Json.parseMap(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

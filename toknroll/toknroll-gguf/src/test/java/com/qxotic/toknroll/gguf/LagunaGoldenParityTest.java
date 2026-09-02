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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Byte-exact parity of the GGUF-loaded Laguna tokenizer against Poolside's reference: every case in
 * {@code laguna_golden_tokens.json} was encoded by Hugging Face {@code tokenizers} from the model's
 * own {@code tokenizer.json} ({@code add_special_tokens=False}, the same contract as {@link
 * Tokenizer#encode}). The tokenizer under test comes from the GGUF's metadata - the {@code laguna}
 * pretokenizer registration, its vocabulary and merges - so the test pins the registration, not
 * just the BPE core. The golden is generated, not committed: {@code
 * toknroll-benchmarks/generate_laguna_golden.py} writes it, and the test skips until it exists.
 *
 * <p>Out of scope by contract: special-token spellings inside the text. {@link Tokenizer#encode} is
 * the non-special-aware path (llama.cpp's default), so it encodes "〈|EOS|〉" as text, where the
 * reference's default would match the added token.
 */
@Tag("network")
@Tag("local-external")
class LagunaGoldenParityTest {

    private static final String CACHE_KEY = "family-laguna-xs-2.1";
    private static final String GGUF_URL =
            "https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/resolve/main/Laguna-XS-2.1-Q4_K_M.gguf";
    private static final String GOLDEN = "laguna_golden_tokens.json";

    private static Tokenizer tokenizer;
    private static Map<String, Object> golden;

    @BeforeAll
    static void load() {
        TestDataManager dataManager = new TestDataManager();
        GGUF gguf;
        try {
            gguf = dataManager.getOrDownloadMetadata(CACHE_KEY, GGUF_URL);
        } catch (Exception e) {
            Assumptions.abort("Laguna GGUF metadata unavailable: " + e);
            return;
        }
        assertEquals("laguna", gguf.getValueOrDefault(String.class, "tokenizer.ggml.pre", "?"));
        Path partial =
                dataManager.getCachePath().resolve(TestDataManager.cacheFileNameForUrl(GGUF_URL));
        try {
            Path local = Files.createTempFile("toknroll-laguna-", ".gguf");
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
        assertEquals("poolside/Laguna-XS-2.1", golden.get("model_ref"));
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

    static Stream<Arguments> cases() {
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

    private static Map<String, Object> readGolden() {
        try (InputStream in = LagunaGoldenParityTest.class.getResourceAsStream("/" + GOLDEN)) {
            if (in == null) {
                Assumptions.abort(
                        GOLDEN
                                + " is not generated; run"
                                + " toknroll-benchmarks/generate_laguna_golden.py");
            }
            return Json.parseMap(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

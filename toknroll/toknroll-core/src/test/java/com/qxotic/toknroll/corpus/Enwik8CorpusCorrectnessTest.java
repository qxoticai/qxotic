package com.qxotic.toknroll.corpus;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.corpus.Enwik8Corpus.Chunk;
import com.qxotic.toknroll.corpus.GroundTruthData.Entry;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.testkit.TestTokenizers;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Comprehensive correctness test using enwik8 corpus chunks against reference tokenizers.
 *
 * <p>This test compares Tok'n'Roll tokenizers against:
 *
 * <ul>
 *   <li>tiktoken (Python) for OpenAI encodings
 *   <li>HuggingFace Transformers for model families
 * </ul>
 *
 * <p>Ground truth data must be generated first by running:
 *
 * <pre>
 *   python benchmarks/generate_enwik8_ground_truth.py
 * </pre>
 *
 * <p>Tests are tagged with "corpus" and "slow" to avoid running in normal CI.
 */
@Tag("corpus")
@Tag("slow")
@TestInstance(Lifecycle.PER_CLASS)
class Enwik8CorpusCorrectnessTest {

    private static final Path GROUND_TRUTH_DIR = Paths.get("src/test/resources/golden/enwik8");

    private static final boolean SKIP_MISSING_GROUND_TRUTH =
            Boolean.parseBoolean(
                    System.getProperty("toknroll.test.skipMissingGroundTruth", "true"));

    private final Map<String, GroundTruthData> groundTruthCache = new ConcurrentHashMap<>();
    private final Map<String, Tokenizer> tokenizerCache = new ConcurrentHashMap<>();
    private List<Chunk> chunks;

    @BeforeAll
    void setUp() {
        // Load chunks from the ground truth chunks.json
        Path chunksPath = GROUND_TRUTH_DIR.resolve("chunks.json");
        if (!Files.exists(chunksPath)) {
            throw new IllegalStateException("chunks.json not found at " + chunksPath);
        }
        chunks = loadChunksFromJson(chunksPath);
        System.out.println("Loaded " + chunks.size() + " chunks from ground truth");
    }

    @SuppressWarnings("unchecked")
    private List<Chunk> loadChunksFromJson(Path path) {
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            List<Map<String, Object>> rawList = (List<Map<String, Object>>) Json.parse(json);
            List<Chunk> result = new ArrayList<>();
            for (Map<String, Object> map : rawList) {
                int offset = ((Number) map.get("offset")).intValue();
                int size = ((Number) map.get("size")).intValue();
                String text = (String) map.get("text");
                result.add(new Chunk(offset, size, text));
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load chunks from " + path, e);
        }
    }

    // ==================== TikToken Encodings ====================

    static Stream<Arguments> tiktokenEncodings() {
        return Stream.of(
                Arguments.of("r50k_base", "tiktoken_r50k_base_ground_truth.json"),
                Arguments.of("cl100k_base", "tiktoken_cl100k_base_ground_truth.json"),
                Arguments.of("o200k_base", "tiktoken_o200k_base_ground_truth.json"));
    }

    @ParameterizedTest(name = "tiktoken {0} encode parity")
    @MethodSource("tiktokenEncodings")
    void tiktokenEncodeParity(String encoding, String groundTruthFile) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Tokenizer tokenizer = getTiktokenTokenizer(encoding);
        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            IntSequence actual = tokenizer.encode(chunk.text());
            int[] expected = entry.tokens();

            if (!Arrays.equals(expected, actual.toArray())) {
                mismatches.add(
                        new Mismatch(
                                chunk, entry, actual.toArray(), "Token mismatch at chunk " + i));
            }
        }

        reportMismatches(encoding + " encode", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                encoding + " has " + mismatches.size() + " encode mismatches");
    }

    @ParameterizedTest(name = "tiktoken {0} decode parity")
    @MethodSource("tiktokenEncodings")
    void tiktokenDecodeParity(String encoding, String groundTruthFile) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Tokenizer tokenizer = getTiktokenTokenizer(encoding);
        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            String actualDecoded = tokenizer.decode(IntSequence.copyOf(entry.tokens()));
            String expectedDecoded = entry.decoded();

            if (!expectedDecoded.equals(actualDecoded)) {
                mismatches.add(
                        new Mismatch(
                                chunk,
                                entry,
                                null,
                                "Decode mismatch at chunk "
                                        + i
                                        + ": expected length "
                                        + expectedDecoded.length()
                                        + " but got "
                                        + actualDecoded.length()));
            }
        }

        reportMismatches(encoding + " decode", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                encoding + " has " + mismatches.size() + " decode mismatches");
    }

    @ParameterizedTest(name = "tiktoken {0} countTokens parity")
    @MethodSource("tiktokenEncodings")
    void tiktokenCountTokensParity(String encoding, String groundTruthFile) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Tokenizer tokenizer = getTiktokenTokenizer(encoding);
        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            int actualCount = tokenizer.countTokens(chunk.text());
            int expectedCount = entry.tokenCount();

            if (expectedCount != actualCount) {
                mismatches.add(
                        new Mismatch(
                                chunk,
                                entry,
                                null,
                                "Count mismatch at chunk "
                                        + i
                                        + ": expected "
                                        + expectedCount
                                        + " but got "
                                        + actualCount));
            }
        }

        reportMismatches(encoding + " countTokens", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                encoding + " has " + mismatches.size() + " countTokens mismatches");
    }

    // ==================== Model Families ====================

    static Stream<Arguments> modelFamilies() {
        return Stream.of(
                Arguments.of(
                        "meta.llama3",
                        "hf_meta_llama3_ground_truth.json",
                        "meta-llama/Llama-3.2-1B-Instruct"),
                Arguments.of(
                        "alibaba.qwen3_5",
                        "hf_alibaba_qwen3_5_ground_truth.json",
                        "Qwen/Qwen3.5-0.8B"),
                Arguments.of(
                        "google.gemma4",
                        "hf_google_gemma4_ground_truth.json",
                        "google/gemma-4-e2b-it"),
                Arguments.of(
                        "huggingface.smollm3",
                        "hf_huggingface_smollm3_ground_truth.json",
                        "HuggingFaceTB/SmolLM3-3B"),
                Arguments.of(
                        "deepseek.v3_2",
                        "hf_deepseek_v3_2_ground_truth.json",
                        "deepseek-ai/DeepSeek-V3.2"),
                Arguments.of(
                        "moonshot.kimi2_5",
                        "hf_moonshot_kimi2_5_ground_truth.json",
                        "moonshotai/Kimi-K2.5"),
                Arguments.of(
                        "mistral.tekken",
                        "hf_mistral_tekken_ground_truth.json",
                        "mistralai/ministral-8b-instruct-2410"),
                Arguments.of(
                        "ibm.granite4_0",
                        "hf_ibm_granite4_0_ground_truth.json",
                        "ibm-granite/granite-4.0-h-1b"),
                Arguments.of(
                        "microsoft.phi4", "hf_microsoft_phi4_ground_truth.json", "microsoft/phi-4"),
                Arguments.of(
                        "mistral.v0_3",
                        "hf_mistral_v0_3_ground_truth.json",
                        "mistralai/Mistral-7B-Instruct-v0.3"));
    }

    @ParameterizedTest(name = "family {0} encode parity")
    @MethodSource("modelFamilies")
    void familyEncodeParity(String familyId, String groundTruthFile, String hfModelRef) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Optional<Tokenizer> maybeTokenizer = TestTokenizers.modelFamily(familyId);
        assumeTrue(maybeTokenizer.isPresent(), "Tokenizer not available for family: " + familyId);
        Tokenizer tokenizer = maybeTokenizer.get();

        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            IntSequence actual = tokenizer.encode(chunk.text());
            int[] expected = entry.tokens();

            if (!Arrays.equals(expected, actual.toArray())) {
                mismatches.add(
                        new Mismatch(
                                chunk, entry, actual.toArray(), "Token mismatch at chunk " + i));
            }
        }

        reportMismatches(familyId + " encode", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                familyId + " has " + mismatches.size() + " encode mismatches");
    }

    @ParameterizedTest(name = "family {0} decode parity")
    @MethodSource("modelFamilies")
    void familyDecodeParity(String familyId, String groundTruthFile, String hfModelRef) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Optional<Tokenizer> maybeTokenizer = TestTokenizers.modelFamily(familyId);
        assumeTrue(maybeTokenizer.isPresent(), "Tokenizer not available for family: " + familyId);
        Tokenizer tokenizer = maybeTokenizer.get();

        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            String actualDecoded = tokenizer.decode(IntSequence.copyOf(entry.tokens()));
            String expectedDecoded = entry.decoded();

            if (!expectedDecoded.equals(actualDecoded)) {
                mismatches.add(new Mismatch(chunk, entry, null, "Decode mismatch at chunk " + i));
            }
        }

        reportMismatches(familyId + " decode", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                familyId + " has " + mismatches.size() + " decode mismatches");
    }

    @ParameterizedTest(name = "family {0} countTokens parity")
    @MethodSource("modelFamilies")
    void familyCountTokensParity(String familyId, String groundTruthFile, String hfModelRef) {
        GroundTruthData gt = loadGroundTruth(groundTruthFile);
        assumeTrue(
                gt != null || !SKIP_MISSING_GROUND_TRUTH,
                "Ground truth not found: " + groundTruthFile);
        if (gt == null) return;

        Optional<Tokenizer> maybeTokenizer = TestTokenizers.modelFamily(familyId);
        assumeTrue(maybeTokenizer.isPresent(), "Tokenizer not available for family: " + familyId);
        Tokenizer tokenizer = maybeTokenizer.get();

        List<Mismatch> mismatches = new ArrayList<>();

        for (int i = 0; i < Math.min(gt.size(), chunks.size()); i++) {
            Entry entry = gt.entries().get(i);
            if (entry.hasError()) continue;

            Chunk chunk = chunks.get(i);
            int actualCount = tokenizer.countTokens(chunk.text());
            int expectedCount = entry.tokenCount();

            if (expectedCount != actualCount) {
                mismatches.add(
                        new Mismatch(
                                chunk,
                                entry,
                                null,
                                "Count mismatch at chunk "
                                        + i
                                        + ": expected "
                                        + expectedCount
                                        + " but got "
                                        + actualCount));
            }
        }

        reportMismatches(familyId + " countTokens", mismatches);
        assertTrue(
                mismatches.isEmpty(),
                familyId + " has " + mismatches.size() + " countTokens mismatches");
    }

    // ==================== Helper Methods ====================

    private GroundTruthData loadGroundTruth(String fileName) {
        return groundTruthCache.computeIfAbsent(
                fileName,
                name -> {
                    Path path = GROUND_TRUTH_DIR.resolve(name);
                    if (!Files.exists(path)) {
                        System.out.println("Ground truth file not found: " + path);
                        return null;
                    }
                    try {
                        return GroundTruthData.load(name, path.toString());
                    } catch (Exception e) {
                        System.out.println(
                                "Failed to load ground truth " + name + ": " + e.getMessage());
                        return null;
                    }
                });
    }

    private Tokenizer getTiktokenTokenizer(String encoding) {
        return tokenizerCache.computeIfAbsent(
                encoding,
                enc -> {
                    // Use our native tokenizer as the test subject
                    // Ground truth comes from Python tiktoken
                    switch (enc) {
                        case "r50k_base":
                            return TestTokenizers.tiktoken("r50k_base", FastSplitters.r50k());
                        case "cl100k_base":
                            return TestTokenizers.tiktoken("cl100k_base", FastSplitters.cl100k());
                        case "o200k_base":
                            return TestTokenizers.tiktoken("o200k_base", FastSplitters.o200k());
                        default:
                            throw new IllegalArgumentException("Unknown encoding: " + enc);
                    }
                });
    }

    private void reportMismatches(String context, List<Mismatch> mismatches) {
        if (mismatches.isEmpty()) return;

        System.err.println("\n=== " + context + " mismatches (" + mismatches.size() + ") ===");
        int limit = Math.min(5, mismatches.size());
        for (int i = 0; i < limit; i++) {
            Mismatch m = mismatches.get(i);
            System.err.println(m.message);
            System.err.println("  Chunk: " + m.chunk);
            if (m.actualTokens != null) {
                System.err.println("  Expected tokens: " + Arrays.toString(m.entry.tokens()));
                System.err.println("  Actual tokens:   " + Arrays.toString(m.actualTokens));
            }
            System.err.println(
                    "  Expected decoded length: "
                            + (m.entry.decoded() != null ? m.entry.decoded().length() : "N/A"));
            System.err.println("  Text preview: " + compact(m.chunk.text()));
        }
        if (mismatches.size() > limit) {
            System.err.println("  ... and " + (mismatches.size() - limit) + " more");
        }
    }

    private String compact(String text) {
        String t = text.replace("\n", "\\n").replace("\t", "\\t");
        return t.length() <= 80 ? t : t.substring(0, 77) + "...";
    }

    private static final class Mismatch {
        final Chunk chunk;
        final Entry entry;
        final int[] actualTokens;
        final String message;

        Mismatch(Chunk chunk, Entry entry, int[] actualTokens, String message) {
            this.chunk = chunk;
            this.entry = entry;
            this.actualTokens = actualTokens;
            this.message = message;
        }
    }
}

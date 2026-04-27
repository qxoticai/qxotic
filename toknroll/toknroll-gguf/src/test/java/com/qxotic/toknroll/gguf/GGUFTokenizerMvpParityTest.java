package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TokenizerInvariantHarness;
import com.qxotic.toknroll.testkit.TokenizerParityHarness;
import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class GGUFTokenizerMvpParityTest {

    private static final Path CORE_GOLDEN_ENWIK8_DIR =
            Path.of("..", "toknroll-core", "src", "test", "resources", "golden", "enwik8");
    private static final int MAX_CHUNKS =
            Integer.getInteger(
                    "toknroll.test.maxChunks",
                    Integer.getInteger("toknroll.test.gguf.maxChunks", 20));
    private static final int CHUNK_SIZE =
            Integer.getInteger("toknroll.test.chunk.size", Integer.MAX_VALUE);
    private static final String CORPUS_PATH_PROPERTY = "toknroll.test.corpus.path";
    private static final String GROUND_TRUTH_SOURCE_PROPERTY =
            "toknroll.test.gguf.groundTruthSource";
    private static final String GROUND_TRUTH_SOURCE =
            System.getProperty(GROUND_TRUTH_SOURCE_PROPERTY, "hf").trim().toLowerCase();
    private static final List<String> INVARIANT_SMOKE_TEXTS =
            List.of(
                    "",
                    "Hello",
                    "Hello world",
                    "Whitespace\\n\\tcheck",
                    "emoji 😀 and accents café",
                    "العربية mixed English 123",
                    "தமிழ் மொழி");

    private static final List<ModelSpec> MODELS =
            List.of(
                    new ModelSpec(
                            "unsloth_llama3_2",
                            "unsloth/Llama-3.2-1B-Instruct-GGUF",
                            "Llama-3.2-1B-Instruct-Q8_0.gguf",
                            "hf_unsloth_llama3_2_ground_truth.json"),
                    new ModelSpec(
                            "google_gemma4",
                            "unsloth/gemma-4-E2B-it-GGUF",
                            "gemma-4-E2B-it-Q8_0.gguf",
                            "hf_google_gemma4_ground_truth.json"),
                    new ModelSpec(
                            "openai_gpt_oss",
                            "unsloth/gpt-oss-20b-GGUF",
                            "gpt-oss-20b-Q8_0.gguf",
                            "hf_openai_gpt_oss_ground_truth.json"),
                    new ModelSpec(
                            "alibaba_qwen3_5",
                            "unsloth/Qwen3.6-35B-A3B-GGUF",
                            "Qwen3.6-35B-A3B-Q8_0.gguf",
                            "hf_alibaba_qwen3_5_ground_truth.json"),
                    new ModelSpec(
                            "microsoft_phi4",
                            "unsloth/phi-4-GGUF",
                            "phi-4-Q8_0.gguf",
                            "hf_microsoft_phi4_ground_truth.json"),
                    new ModelSpec(
                            "ibm_granite4_0",
                            "unsloth/granite-4.0-h-1b-GGUF",
                            "granite-4.0-h-1b-Q8_0.gguf",
                            "hf_ibm_granite4_0_ground_truth.json"),
                    new ModelSpec(
                            "huggingface_smollm3",
                            "unsloth/SmolLM3-3B-GGUF",
                            "SmolLM3-3B-Q8_0.gguf",
                            "hf_huggingface_smollm3_ground_truth.json"),
                    new ModelSpec(
                            "moonshot_kimi2_6",
                            "unsloth/Kimi-K2.6-GGUF",
                            "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf",
                            "hf_moonshot_kimi2_6_ground_truth.json"),
                    new ModelSpec(
                            "zai_glm5_1",
                            "unsloth/GLM-5.1-GGUF",
                            "Q8_0/GLM-5.1-Q8_0-00001-of-00017.gguf",
                            "hf_zai_glm5_1_ground_truth.json"),
                    new ModelSpec(
                            "minimax_m2_7",
                            "unsloth/MiniMax-M2.7-GGUF",
                            "Q8_0/MiniMax-M2.7-Q8_0-00001-of-00006.gguf",
                            "hf_minimax_m2_7_ground_truth.json"),
                    new ModelSpec(
                            "xiaomi_mimo_v2_flash",
                            "unsloth/MiMo-V2-Flash-GGUF",
                            "Q8_0/MiMo-V2-Flash-Q8_0-00001-of-00007.gguf",
                            "hf_xiaomi_mimo_v2_flash_ground_truth.json"),
                    new ModelSpec(
                            "mistral_mistral7b_v0_3",
                            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                            "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                            "hf_mistral_mistral7b_v0_3_ground_truth.json"));

    @Test
    void parityOnMvpGgufModels() throws Exception {
        assertTrue(
                Files.isDirectory(CORE_GOLDEN_ENWIK8_DIR),
                "Missing golden dir: " + CORE_GOLDEN_ENWIK8_DIR);
        byte[] corpusBytes = loadCorpusBytes();

        GGUFTokenizerLoader loader = GGUFTokenizerLoader.builderDefault().build();
        for (ModelSpec modelSpec : MODELS) {
            if ("llamacpp".equals(GROUND_TRUTH_SOURCE)
                    && "google_gemma4".equals(modelSpec.familyId)) {
                continue;
            }
            Tokenizer tokenizer =
                    loader.fromHuggingFace(
                            modelSpec.user, modelSpec.repository, modelSpec.ggufPath);
            assertSmokeInvariants(modelSpec.modelRef, tokenizer);

            Path goldenPath =
                    CORE_GOLDEN_ENWIK8_DIR
                            .resolve(modelSpec.groundTruthFileName(GROUND_TRUTH_SOURCE))
                            .toAbsolutePath()
                            .normalize();
            assertTrue(Files.exists(goldenPath), "Missing golden file: " + goldenPath);

            System.err.println("Start " + modelSpec.ggufPath);
            TokenizerParityHarness.runParity(
                    modelSpec.modelRef,
                    goldenPath,
                    corpusBytes,
                    CHUNK_SIZE,
                    MAX_CHUNKS,
                    text -> tokenizer.encode(text).toArray(),
                    tokenizer::countTokens,
                    tokens -> tokenizer.decode(IntSequence.wrap(tokens)),
                    tokenId -> tokenizer.vocabulary().token(tokenId));
            System.err.println("Done " + modelSpec.ggufPath);
        }
    }

    private static byte[] loadCorpusBytes() throws Exception {
        String corpusPath = System.getProperty(CORPUS_PATH_PROPERTY);
        if (corpusPath != null && !corpusPath.isBlank()) {
            return TokenizerParityHarness.loadCorpusBytes(Path.of(corpusPath));
        }
        return Enwik8Corpus.load().data();
    }

    private static void assertSmokeInvariants(String label, Tokenizer tokenizer) {
        TokenizerInvariantHarness.runSmokeChecks(
                label,
                INVARIANT_SMOKE_TEXTS,
                text -> tokenizer.encode(text).toArray(),
                text -> {
                    IntSequence.Builder out = IntSequence.newBuilder(16);
                    tokenizer.encodeInto(text, out);
                    return out.build().toArray();
                },
                tokenizer::countTokens,
                tokens -> tokenizer.decode(IntSequence.wrap(tokens)),
                tokens -> tokenizer.decodeBytes(IntSequence.wrap(tokens)),
                tokens -> tokenizer.countBytes(IntSequence.wrap(tokens)),
                (tokens, tokenStartIndex, out) ->
                        tokenizer.decodeBytesInto(IntSequence.wrap(tokens), tokenStartIndex, out));
    }

    private static final class ModelSpec {
        private final String familyId;
        private final String modelRef;
        private final String user;
        private final String repository;
        private final String ggufPath;
        private final String groundTruthFileName;

        private ModelSpec(
                String familyId, String modelRef, String ggufPath, String hfGroundTruthFileName) {
            this.familyId = familyId;
            this.modelRef = modelRef;
            String[] parts = modelRef.split("/", 2);
            if (parts.length != 2) {
                throw new IllegalArgumentException(
                        "Model ref must be user/repository: " + modelRef);
            }
            this.user = parts[0];
            this.repository = parts[1];
            this.ggufPath = ggufPath;
            this.groundTruthFileName = hfGroundTruthFileName;
        }

        private String groundTruthFileName(String source) {
            if ("llamacpp".equals(source)) {
                return "llamacpp_" + familyId + "_ground_truth.json";
            }
            return groundTruthFileName;
        }
    }
}

package com.qxotic.toknroll.hf;

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
class HuggingFaceTokenizerParityTest {

    private static final Path CORE_GOLDEN_ENWIK8_DIR =
            Path.of("..", "toknroll-core", "src", "test", "resources", "golden", "enwik8");
    private static final int MAX_CHUNKS =
            Integer.getInteger(
                    "toknroll.test.maxChunks",
                    Integer.getInteger("toknroll.test.hf.maxChunks", 20));
    private static final int CHUNK_SIZE =
            Integer.getInteger("toknroll.test.chunk.size", Integer.MAX_VALUE);
    private static final String CORPUS_PATH_PROPERTY = "toknroll.test.corpus.path";
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
                    new ModelSpec("Qwen/Qwen3.5-0.8B", "hf_alibaba_qwen3_5_ground_truth.json"),
                    new ModelSpec(
                            "unsloth/Llama-3.2-1B-Instruct",
                            "hf_unsloth_llama3_2_ground_truth.json"),
                    new ModelSpec(
                            "deepseek-ai/DeepSeek-V3.2", "hf_deepseek_v3_2_ground_truth.json"),
                    new ModelSpec(
                            "deepseek-ai/DeepSeek-V4-Pro", "hf_deepseek_v4_pro_ground_truth.json"),
                    new ModelSpec("moonshotai/Kimi-K2.6", "hf_moonshot_kimi2_6_ground_truth.json"),
                    new ModelSpec(
                            "HuggingFaceTB/SmolLM3-3B", "hf_huggingface_smollm3_ground_truth.json"),
                    new ModelSpec(
                            "ibm-granite/granite-4.1-8b", "hf_ibm_granite4_1_8b_ground_truth.json"),
                    new ModelSpec(
                            "mistralai/ministral-8b-instruct-2410",
                            "hf_mistral_tekken_ground_truth.json"),
                    new ModelSpec("zai-org/GLM-5.1", "hf_zai_glm5_1_ground_truth.json"),
                    new ModelSpec("MiniMaxAI/MiniMax-M2.7", "hf_minimax_m2_7_ground_truth.json"),
                    new ModelSpec(
                            "XiaomiMiMo/MiMo-V2-Flash",
                            "hf_xiaomi_mimo_v2_flash_ground_truth.json"),
                    new ModelSpec("microsoft/phi-4", "hf_microsoft_phi4_ground_truth.json"),
                    new ModelSpec("openai/gpt-oss-20b", "hf_openai_gpt_oss_ground_truth.json"),
                    new ModelSpec("google/gemma-4-e2b-it", "hf_google_gemma4_ground_truth.json"));

    @Test
    void parityOnModels() throws Exception {
        assertTrue(
                Files.isDirectory(CORE_GOLDEN_ENWIK8_DIR),
                "Missing golden dir: " + CORE_GOLDEN_ENWIK8_DIR);
        byte[] corpusBytes = loadCorpusBytes();

        for (ModelSpec modelSpec : MODELS) {
            System.err.println("Loading " + modelSpec.modelRef);
            Tokenizer tokenizer =
                    HuggingFaceTokenizerLoader.fromHuggingFace(
                            modelSpec.user, modelSpec.repository, "main", false, false);
            System.err.println("Testing invariants for " + modelSpec.modelRef);
            assertSmokeInvariants(modelSpec.modelRef, tokenizer);

            Path goldenPath =
                    CORE_GOLDEN_ENWIK8_DIR
                            .resolve(modelSpec.groundTruthFileName)
                            .toAbsolutePath()
                            .normalize();
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
        private final String modelRef;
        private final String user;
        private final String repository;
        private final String groundTruthFileName;

        private ModelSpec(String modelRef, String groundTruthFileName) {
            this.modelRef = modelRef;
            String[] parts = modelRef.split("/", 2);
            if (parts.length != 2) {
                throw new IllegalArgumentException(
                        "Model ref must be user/repository: " + modelRef);
            }
            this.user = parts[0];
            this.repository = parts[1];
            this.groundTruthFileName = groundTruthFileName;
        }
    }
}

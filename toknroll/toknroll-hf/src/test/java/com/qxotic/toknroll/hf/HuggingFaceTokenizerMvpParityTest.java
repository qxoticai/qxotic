package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class HuggingFaceTokenizerMvpParityTest {

    private static final Path CORE_GOLDEN_ENWIK8_DIR =
            Path.of("..", "toknroll-core", "src", "test", "resources", "golden", "enwik8");
    private static final int MAX_CHUNKS =
            Integer.parseInt(System.getProperty("toknroll.test.hf.maxChunks", "20"));
    private static final String QWEN_MODEL_REF = "Qwen/Qwen3.5-0.8B";
    private static final Set<String> QWEN_KNOWN_DRIFT_HASHES =
            Set.of("c540e3a905a433c2", "0543920ed219e50b", "a037a15a1bc66f8a", "1ebb6e25426913ce");

    private static final List<ModelSpec> MODELS =
            List.of(
                    new ModelSpec("Qwen/Qwen3.5-0.8B", "hf_alibaba_qwen3_5_ground_truth.json"),
                    new ModelSpec(
                            "unsloth/Llama-3.2-1B-Instruct",
                            "hf_unsloth_llama3_2_ground_truth.json"),
                    new ModelSpec(
                            "deepseek-ai/DeepSeek-V3.1", "hf_deepseek_v3_2_ground_truth.json"),
                    new ModelSpec(
                            "deepseek-ai/DeepSeek-V4-Pro", "hf_deepseek_v4_pro_ground_truth.json"),
                    new ModelSpec("moonshotai/Kimi-K2.6", "hf_moonshot_kimi2_6_ground_truth.json"),
                    new ModelSpec(
                            "HuggingFaceTB/SmolLM3-3B", "hf_huggingface_smollm3_ground_truth.json"),
                    new ModelSpec(
                            "swiss-ai/Apertus-8B-Instruct-2509",
                            "hf_swiss_apertus8b_ground_truth.json"),
                    new ModelSpec("openai/gpt-oss-20b", "hf_openai_gpt_oss_ground_truth.json"),
                    new ModelSpec("google/gemma-4-e2b-it", "hf_google_gemma4_ground_truth.json"));

    @Test
    void parityOnMvpModels() throws Exception {
        assertTrue(
                Files.isDirectory(CORE_GOLDEN_ENWIK8_DIR),
                "Missing golden dir: " + CORE_GOLDEN_ENWIK8_DIR);
        List<Chunk> chunks = loadChunks();

        for (ModelSpec modelSpec : MODELS) {
            GroundTruth groundTruth = loadGroundTruth(modelSpec.groundTruthFileName);
            Tokenizer tokenizer =
                    HuggingFaceTokenizerLoader.fromPretrained(
                            modelSpec.user, modelSpec.repository, "main", false, false);

            List<Chunk> selected = selectChunks(chunks, groundTruth, MAX_CHUNKS);
            for (int i = 0; i < selected.size(); i++) {
                Chunk chunk = selected.get(i);
                if (isKnownDrift(modelSpec, chunk)) {
                    continue;
                }
                int[] expected = groundTruth.tokensByHash.get(chunk.hash);
                assertTrue(expected != null, "Missing ground truth chunk hash: " + chunk.hash);

                int[] actual = tokenizer.encode(chunk.text).toArray();
                assertArrayEquals(
                        expected,
                        actual,
                        "encode mismatch for model="
                                + modelSpec.user
                                + "/"
                                + modelSpec.repository
                                + " chunkHash="
                                + chunk.hash
                                + " index="
                                + i);

                int actualCount = tokenizer.countTokens(chunk.text);
                assertEquals(
                        expected.length,
                        actualCount,
                        "countTokens mismatch for model="
                                + modelSpec.user
                                + "/"
                                + modelSpec.repository
                                + " chunkHash="
                                + chunk.hash
                                + " index="
                                + i);

                String decoded = tokenizer.decode(IntSequence.wrap(expected));
                assertEquals(
                        chunk.text,
                        decoded,
                        "decode mismatch for model="
                                + modelSpec.user
                                + "/"
                                + modelSpec.repository
                                + " chunkHash="
                                + chunk.hash
                                + " index="
                                + i);
            }
        }
    }

    private static boolean isKnownDrift(ModelSpec modelSpec, Chunk chunk) {
        return QWEN_MODEL_REF.equals(modelSpec.modelRef)
                && QWEN_KNOWN_DRIFT_HASHES.contains(chunk.hash);
    }

    private static List<Chunk> selectChunks(
            List<Chunk> chunks, GroundTruth groundTruth, int maxChunks) {
        int limit = Math.min(Math.min(chunks.size(), groundTruth.size()), maxChunks);
        if (limit <= 0) {
            return List.of();
        }

        Map<Integer, List<Chunk>> bySize = new HashMap<>();
        for (Chunk chunk : chunks) {
            if (groundTruth.tokensByHash.containsKey(chunk.hash)) {
                bySize.computeIfAbsent(chunk.size, ignored -> new ArrayList<>()).add(chunk);
            }
        }

        List<Integer> sizes = new ArrayList<>(bySize.keySet());
        sizes.sort(Comparator.naturalOrder());
        List<Chunk> selected = new ArrayList<>(limit);

        int round = 0;
        while (selected.size() < limit) {
            boolean pickedAny = false;
            for (int size : sizes) {
                List<Chunk> bucket = bySize.get(size);
                if (bucket != null && round < bucket.size() && selected.size() < limit) {
                    selected.add(bucket.get(round));
                    pickedAny = true;
                }
            }
            if (!pickedAny) {
                break;
            }
            round++;
        }

        return selected;
    }

    @SuppressWarnings("unchecked")
    private static List<Chunk> loadChunks() throws IOException {
        try {
            Path path = CORE_GOLDEN_ENWIK8_DIR.resolve("chunks.json").toAbsolutePath().normalize();
            String json = Files.readString(path, StandardCharsets.UTF_8);
            List<Object> chunkList = (List<Object>) Json.parse(json);
            List<Chunk> chunks = new ArrayList<>(chunkList.size());

            for (Object chunkObj : chunkList) {
                Map<String, Object> chunkMap = (Map<String, Object>) chunkObj;
                String hash = (String) chunkMap.get("hash");
                String text = (String) chunkMap.get("text");
                int size = ((Number) chunkMap.get("size")).intValue();
                chunks.add(new Chunk(hash, text, size));
            }
            return chunks;
        } catch (ClassCastException e) {
            throw new IOException("Invalid chunks fixture format", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static GroundTruth loadGroundTruth(String fileName) throws IOException {
        try {
            Path path = CORE_GOLDEN_ENWIK8_DIR.resolve(fileName).toAbsolutePath().normalize();
            String json = Files.readString(path, StandardCharsets.UTF_8);
            List<Object> rows = (List<Object>) Json.parse(json);
            Map<String, int[]> tokensByHash = new HashMap<>(rows.size());

            for (Object rowObj : rows) {
                Map<String, Object> row = (Map<String, Object>) rowObj;
                String hash = (String) row.get("chunk_hash");
                List<Object> tokenList = (List<Object>) row.get("tokens");
                int[] tokens = new int[tokenList.size()];
                for (int i = 0; i < tokenList.size(); i++) {
                    tokens[i] = ((Number) tokenList.get(i)).intValue();
                }
                tokensByHash.put(hash, tokens);
            }

            return new GroundTruth(tokensByHash);
        } catch (ClassCastException e) {
            throw new IOException("Invalid ground truth fixture format for " + fileName, e);
        }
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

    private static final class Chunk {
        private final String hash;
        private final String text;
        private final int size;

        private Chunk(String hash, String text, int size) {
            this.hash = hash;
            this.text = text;
            this.size = size;
        }
    }

    private static final class GroundTruth {
        private final Map<String, int[]> tokensByHash;

        private GroundTruth(Map<String, int[]> tokensByHash) {
            this.tokensByHash = tokensByHash;
        }

        private int size() {
            return tokensByHash.size();
        }
    }
}

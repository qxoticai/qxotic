package com.qxotic.jinfer.testkit;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;

/**
 * The one place test-model paths live. Every model-gated test resolves its GGUF here as {@code
 * ROOT/{source}/{user}/{repo}/{file}} ({@code hf.co}, {@code modelscope.cn}, ...), mirroring where
 * the file came from, so {@code scripts/download-models.sh} can populate the tree from the same
 * manifest ({@code scripts/models.txt} - a unit test keeps fixture and manifest identical).
 *
 * <p>ROOT: {@code -Djinfer.models} &gt; {@code $JINFER_MODELS} &gt; the {@code models} directory
 * next to the git checkout (walk up from the working directory to {@code .git}, then its parent's
 * {@code models}).
 */
public final class ModelFixture {

    private ModelFixture() {}

    public static final Path ROOT = resolveRoot();

    /** One test model: where it came from and where it lives under {@link #ROOT}. */
    public record Gguf(String source, String user, String repo, String file) {

        public Path path() {
            return ROOT.resolve(source).resolve(user).resolve(repo).resolve(file);
        }

        public boolean present() {
            return Files.exists(path());
        }

        /** The path, assume-skipping the test when the file is absent (with the fix in hand). */
        public Path require() {
            Assumptions.assumeTrue(
                    present(), "model not found: " + path() + " - run scripts/download-models.sh");
            return path();
        }
    }

    private static final List<Gguf> ALL = new ArrayList<>();

    private static Gguf hf(String user, String repo, String file) {
        Gguf g = new Gguf("hf.co", user, repo, file);
        ALL.add(g);
        return g;
    }

    // ---- the models (quant in the name; sidecars share their model's repo) ----

    public static final Gguf LFM25_8B_Q8 =
            hf("LiquidAI", "LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q8_0.gguf");
    public static final Gguf LFM2_8B_A1B_Q8 =
            hf("LiquidAI", "LFM2-8B-A1B-GGUF", "LFM2-8B-A1B-Q8_0.gguf");
    public static final Gguf LFM25_26B_Q8 =
            hf("LiquidAI", "LFM2.5-2.6B-GGUF", "LFM2.5-2.6B-Q8_0.gguf");
    public static final Gguf LFM25_350M_Q8 =
            hf("LiquidAI", "LFM2.5-350M-GGUF", "LFM2.5-350M-Q8_0.gguf");
    public static final Gguf LFM25_EMBEDDING_350M_Q8 =
            hf("LiquidAI", "LFM2.5-Embedding-350M-GGUF", "LFM2.5-Embedding-350M-Q8_0.gguf");
    public static final Gguf LFM25_COLBERT_350M_Q8 =
            hf("LiquidAI", "LFM2.5-ColBERT-350M-GGUF", "LFM2.5-ColBERT-350M-Q8_0.gguf");
    public static final Gguf LFM25_VL_3B_Q4 =
            hf("LiquidAI", "LFM2.5-VL-3B-GGUF", "LFM2.5-VL-3B-Q4_K_M.gguf");
    public static final Gguf LFM25_VL_3B_MMPROJ_Q8 =
            hf("LiquidAI", "LFM2.5-VL-3B-GGUF", "mmproj-LFM2.5-VL-3B-Q8_0.gguf");

    public static final Gguf GEMMA4_E2B_Q8 =
            hf("unsloth", "gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf");
    public static final Gguf GEMMA4_E2B_MMPROJ =
            hf("unsloth", "gemma-4-E2B-it-GGUF", "mmproj-F32.gguf");
    public static final Gguf GEMMA4_E2B_MTP =
            hf("unsloth", "gemma-4-E2B-it-GGUF", "mtp-gemma-4-E2B-it.gguf");
    public static final Gguf GEMMA4_E2B_QAT_Q4 =
            hf("unsloth", "gemma-4-E2B-it-qat-GGUF", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
    public static final Gguf GEMMA4_E4B_Q8 =
            hf("unsloth", "gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q8_0.gguf");
    public static final Gguf GEMMA4_E2B_GOOGLE_Q4 =
            hf("google", "gemma-4-E2B-it-qat-q4_0-gguf", "gemma-4-E2B_q4_0-it.gguf");
    public static final Gguf GEMMA4_12B_Q8 =
            hf("unsloth", "gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q8_0.gguf");
    public static final Gguf GEMMA4_12B_MMPROJ =
            hf("unsloth", "gemma-4-12b-it-GGUF", "mmproj-F32.gguf");
    public static final Gguf GEMMA4_12B_QAT_Q4 =
            hf("unsloth", "gemma-4-12B-it-qat-GGUF", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf");
    public static final Gguf GEMMA4_12B_QAT_MMPROJ =
            hf("unsloth", "gemma-4-12B-it-qat-GGUF", "mmproj-F32.gguf");
    public static final Gguf GEMMA4_26B_MOE_Q8 =
            hf("unsloth", "gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-Q8_0.gguf");

    public static final Gguf QWEN35_2B_Q8 = hf("unsloth", "Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q8_0.gguf");
    public static final Gguf QWEN35_4B_Q8 = hf("unsloth", "Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf");

    public static final Gguf GPTOSS_20B_Q8 =
            hf("unsloth", "gpt-oss-20b-GGUF", "gpt-oss-20b-Q8_0.gguf");

    public static final Gguf MAPLE_PREVIEW_TQ1_Q4_HEAD =
            hf("deepgrove", "maple-preview-GGUF", "maple-preview-TQ1_0-head-Q4_K.gguf");

    public static final Gguf LLAMA32_1B_Q8 =
            hf("unsloth", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");

    public static final Gguf MINISTRAL_3B_Q8 =
            hf("unsloth", "Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q8_0.gguf");

    public static final Gguf MINICPM5_1B_Q8 =
            hf("openbmb", "MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q8_0.gguf");

    public static final Gguf GRANITE_41_3B_Q8 =
            hf("ibm-granite", "granite-4.1-3b-GGUF", "granite-4.1-3b-Q8_0.gguf");

    public static final Gguf SMOLLM3_Q4 = hf("ggml-org", "SmolLM3-3B-GGUF", "SmolLM3-Q4_K_M.gguf");

    public static final Gguf NEMOTRON_30B_Q8 =
            hf(
                    "bartowski",
                    "nvidia_Nemotron-Cascade-2-30B-A3B-GGUF",
                    "nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf");

    public static final Gguf QWEN3_RERANKER_06B_Q8 =
            hf("mradermacher", "Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B.Q8_0.gguf");

    public static final Gguf QWEN3_EMBED_06B_Q8 =
            hf("Qwen", "Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf");

    public static final Gguf BONSAI_27B_Q1 =
            hf("prism-ml", "Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf");

    // Speech. The lexicon is a fixture in its own right, not a sidecar of the GGUF: it is a
    // separate file in the same repo, the port looks for it BESIDE the model, and a checkout with
    // the GGUF but no lexicon takes a different (espeak) code path - so a test that needs the
    // lexicon path must be able to say so and skip when it is missing.
    public static final Gguf INFLECT_NANO_V2_Q8 =
            hf("remixerdec", "Inflect-Nano-v2-GGUF", "inflect_nano_v2_q8_0.gguf");
    public static final Gguf INFLECT_NANO_V2_LEXICON =
            hf("remixerdec", "Inflect-Nano-v2-GGUF", "lexicon.bin");
    public static final Gguf INFLECT_MICRO_V2_Q8 =
            hf("remixerdec", "Inflect-Micro-v2-GGUF", "inflect_micro_v2_q8_0.gguf");
    public static final Gguf INFLECT_MICRO_V2_LEXICON =
            hf("remixerdec", "Inflect-Micro-v2-GGUF", "lexicon.bin");

    /** Every declared model, for the manifest-consistency check. */
    public static List<Gguf> all() {
        return List.copyOf(ALL);
    }

    private static Path resolveRoot() {
        String prop = System.getProperty("jinfer.models");
        if (prop != null && !prop.isBlank()) return Path.of(prop);
        String env = System.getenv("JINFER_MODELS");
        if (env != null && !env.isBlank()) return Path.of(env);
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            if (Files.exists(dir.resolve(".git"))) { // a FILE in worktrees, a directory otherwise
                return dir.getParent() == null ? dir.resolve("models") : dir.getParent().resolve("models");
            }
        }
        return Path.of(System.getProperty("user.home"), "models");
    }
}

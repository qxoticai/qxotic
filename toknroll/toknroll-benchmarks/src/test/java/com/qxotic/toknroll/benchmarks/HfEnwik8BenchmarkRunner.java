package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Simple enwik8 throughput runner for common Tok'n'Roll HF model families.
 *
 * <p>Usage:
 *
 * <pre>
 * mvnd -f toknroll/pom.xml -pl toknroll-benchmarks \
 *   -Dexec.mainClass=com.qxotic.toknroll.benchmarks.HfEnwik8BenchmarkRunner \
 *   -Dexec.classpathScope=test exec:java
 * </pre>
 */
public final class HfEnwik8BenchmarkRunner {

    private static final List<ModelSpec> COMMON_MODELS =
            List.of(
                    new ModelSpec("meta.llama3", "unsloth", "Llama-3.2-1B-Instruct", "main"),
                    new ModelSpec("alibaba.qwen3_5", "Qwen", "Qwen3.5-0.8B", "main"),
                    new ModelSpec("google.gemma4", "google", "gemma-4-e2b-it", "main"),
                    new ModelSpec("huggingface.smollm3", "HuggingFaceTB", "SmolLM3-3B", "main"),
                    new ModelSpec("microsoft.phi4", "microsoft", "phi-4", "main"),
                    new ModelSpec("deepseek.v3_2", "deepseek-ai", "DeepSeek-V3.2", "main"),
                    new ModelSpec("moonshot.kimi2_6", "moonshotai", "Kimi-K2.6", "main"),
                    new ModelSpec(
                            "mistral.v0_3_spbpe", "mistralai", "Mistral-7B-Instruct-v0.3", "main"),
                    new ModelSpec("openai.gpt-oss", "openai", "gpt-oss-20b", "main"));

    private HfEnwik8BenchmarkRunner() {}

    public static void main(String[] args) throws IOException {
        String corpus = loadEnwik8();
        String warmupText = corpus.substring(0, Math.min(corpus.length(), 262_144));
        double corpusMb = corpus.length() / 1_000_000.0;

        System.out.println(
                "family,model_ref,chars,tokens,encode_s,encode_mb_s,count_s,count_mtoks_s");
        for (ModelSpec spec : COMMON_MODELS) {
            String modelRef = spec.user + "/" + spec.repo;
            try {
                Tokenizer tokenizer =
                        HuggingFaceTokenizerLoader.fromPretrained(
                                spec.user, spec.repo, spec.revision, false, false);

                tokenizer.encode(warmupText);
                tokenizer.countTokens(warmupText);

                long t0 = System.nanoTime();
                IntSequence encoded = tokenizer.encode(corpus);
                long t1 = System.nanoTime();
                int count = tokenizer.countTokens(corpus);
                long t2 = System.nanoTime();

                double encodeSec = (t1 - t0) / 1_000_000_000.0;
                double countSec = (t2 - t1) / 1_000_000_000.0;
                double encodeMbPerSec = corpusMb / encodeSec;
                double countMtoksPerSec = (count / 1_000_000.0) / countSec;

                System.out.printf(
                        "%s,%s,%d,%d,%.6f,%.3f,%.6f,%.3f%n",
                        spec.familyId,
                        modelRef,
                        corpus.length(),
                        encoded.length(),
                        encodeSec,
                        encodeMbPerSec,
                        countSec,
                        countMtoksPerSec);
            } catch (RuntimeException e) {
                System.out.printf("%s,%s,ERROR,%s%n", spec.familyId, modelRef, summarizeError(e));
            }
        }
    }

    private static String loadEnwik8() throws IOException {
        Path path = WikiCorpusPaths.forCorpus("enwik8");
        return Files.readString(path);
    }

    private static String summarizeError(RuntimeException e) {
        String message = e.getMessage();
        if (message == null || message.isBlank()) {
            return e.getClass().getSimpleName();
        }
        return e.getClass().getSimpleName() + ":" + message.replace(',', ';');
    }

    private static final class ModelSpec {
        private final String familyId;
        private final String user;
        private final String repo;
        private final String revision;

        private ModelSpec(String familyId, String user, String repo, String revision) {
            this.familyId = familyId;
            this.user = user;
            this.repo = repo;
            this.revision = revision;
        }
    }
}

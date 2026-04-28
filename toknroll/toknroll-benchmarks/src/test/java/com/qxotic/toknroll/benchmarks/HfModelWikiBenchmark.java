package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** JMH benchmark for Tok'n'Roll HF-backed tokenizers on enwik corpora. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class HfModelWikiBenchmark {

    private static final Map<String, ModelSpec> MODEL_SPECS =
            Map.ofEntries(
                    Map.entry("openai.gpt-oss", new ModelSpec("openai", "gpt-oss-20b", "main")),
                    Map.entry(
                            "meta.llama3",
                            new ModelSpec("unsloth", "Llama-3.2-1B-Instruct", "main")),
                    Map.entry("moonshot.kimi2_6", new ModelSpec("moonshotai", "Kimi-K2.6", "main")),
                    Map.entry(
                            "huggingface.smollm3",
                            new ModelSpec("HuggingFaceTB", "SmolLM3-3B", "main")),
                    Map.entry("alibaba.qwen3_5", new ModelSpec("Qwen", "Qwen3.5-0.8B", "main")),
                    Map.entry("google.gemma4", new ModelSpec("google", "gemma-4-e2b-it", "main")),
                    Map.entry(
                            "ibm.granite4_0",
                            new ModelSpec("ibm-granite", "granite-4.0-h-1b", "main")),
                    Map.entry("microsoft.phi4", new ModelSpec("microsoft", "phi-4", "main")),
                    Map.entry(
                            "deepseek.v3_2", new ModelSpec("deepseek-ai", "DeepSeek-V3.2", "main")),
                    Map.entry(
                            "mistral.v0_3_spbpe",
                            new ModelSpec("mistralai", "Mistral-7B-Instruct-v0.3", "main")));

    private static final Set<String> PARAM_MODEL_FAMILIES =
            Set.of(
                    "openai.gpt-oss",
                    "meta.llama3",
                    "moonshot.kimi2_6",
                    "huggingface.smollm3",
                    "alibaba.qwen3_5",
                    "google.gemma4",
                    "ibm.granite4_0",
                    "microsoft.phi4",
                    "deepseek.v3_2",
                    "mistral.v0_3_spbpe");

    // JMH @Param values are annotation literals, so keep this explicit set and fail fast if it
    // drifts from MODEL_SPECS.
    static {
        if (!new LinkedHashSet<>(MODEL_SPECS.keySet())
                .equals(new LinkedHashSet<>(PARAM_MODEL_FAMILIES))) {
            throw new IllegalStateException(
                    "HfModelWikiBenchmark model map and @Param values drifted. map="
                            + MODEL_SPECS.keySet()
                            + " param="
                            + PARAM_MODEL_FAMILIES);
        }
    }

    @Param({"enwik8"})
    public String corpus;

    @Param({"1", "4"})
    public int sliceMiB;

    @Param({
        "openai.gpt-oss",
        "meta.llama3",
        "moonshot.kimi2_6",
        "huggingface.smollm3",
        "alibaba.qwen3_5",
        "google.gemma4",
        "ibm.granite4_0",
        "microsoft.phi4",
        "deepseek.v3_2",
        "mistral.v0_3_spbpe"
    })
    public String modelFamily;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            text = WikiBenchmarkSupport.loadCorpusSlice(corpus, sliceMiB);
            tokenizer = loadTokenizer(modelFamily);
            encoded = tokenizer.encode(text);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to initialize HF wiki benchmark for " + modelFamily, e);
        }
    }

    @Benchmark
    public void encode(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text));
    }

    @Benchmark
    public void decode(Blackhole blackhole) {
        blackhole.consume(tokenizer.decode(encoded));
    }

    @Benchmark
    public void countTokens(Blackhole blackhole) {
        blackhole.consume(tokenizer.countTokens(text));
    }

    private static Tokenizer loadTokenizer(String familyId) {
        ModelSpec spec = MODEL_SPECS.get(familyId);
        if (spec == null) {
            throw new IllegalArgumentException(
                    "Unsupported HF model family: "
                            + familyId
                            + ". Supported: "
                            + MODEL_SPECS.keySet().stream()
                                    .sorted()
                                    .collect(Collectors.joining(", ")));
        }
        return HuggingFaceTokenizerLoader.fromHuggingFace(
                spec.user(), spec.repository(), spec.revision(), false, false);
    }

    private record ModelSpec(String user, String repository, String revision) {}
}

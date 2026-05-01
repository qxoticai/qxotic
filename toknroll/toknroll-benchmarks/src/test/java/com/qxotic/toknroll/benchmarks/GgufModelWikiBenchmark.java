package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
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

/** JMH benchmark for Tok'n'Roll GGUF-backed tokenizers on enwik corpora. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class GgufModelWikiBenchmark {

    private static final GGUFTokenizerLoader LOADER =
            GGUFTokenizerLoader.createBuilderWithBuiltins().build();
    private static final Map<String, ModelSpec> MODEL_SPECS =
            Map.ofEntries(
                    Map.entry(
                            "unsloth_llama3_2",
                            new ModelSpec(
                                    "unsloth",
                                    "Llama-3.2-1B-Instruct-GGUF",
                                    "Llama-3.2-1B-Instruct-Q8_0.gguf")),
                    Map.entry(
                            "google_gemma4",
                            new ModelSpec(
                                    "unsloth", "gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf")),
                    Map.entry(
                            "openai_gpt_oss",
                            new ModelSpec("unsloth", "gpt-oss-20b-GGUF", "gpt-oss-20b-Q8_0.gguf")),
                    Map.entry(
                            "alibaba_qwen3_5",
                            new ModelSpec(
                                    "unsloth",
                                    "Qwen3.6-35B-A3B-GGUF",
                                    "Qwen3.6-35B-A3B-Q8_0.gguf")),
                    Map.entry(
                            "microsoft_phi4",
                            new ModelSpec("unsloth", "phi-4-GGUF", "phi-4-Q8_0.gguf")),
                    Map.entry(
                            "ibm_granite4_1_3b",
                            new ModelSpec(
                                    "unsloth", "granite-4.1-3b-GGUF", "granite-4.1-3b-Q8_0.gguf")),
                    Map.entry(
                            "huggingface_smollm3",
                            new ModelSpec("unsloth", "SmolLM3-3B-GGUF", "SmolLM3-3B-Q8_0.gguf")),
                    Map.entry(
                            "moonshot_kimi2_6",
                            new ModelSpec(
                                    "unsloth",
                                    "Kimi-K2.6-GGUF",
                                    "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf")),
                    Map.entry(
                            "zai_glm5_1",
                            new ModelSpec(
                                    "unsloth",
                                    "GLM-5.1-GGUF",
                                    "Q8_0/GLM-5.1-Q8_0-00001-of-00017.gguf")),
                    Map.entry(
                            "minimax_m2_7",
                            new ModelSpec(
                                    "unsloth",
                                    "MiniMax-M2.7-GGUF",
                                    "Q8_0/MiniMax-M2.7-Q8_0-00001-of-00006.gguf")),
                    Map.entry(
                            "xiaomi_mimo_v2_flash",
                            new ModelSpec(
                                    "unsloth",
                                    "MiMo-V2-Flash-GGUF",
                                    "Q8_0/MiMo-V2-Flash-Q8_0-00001-of-00007.gguf")),
                    Map.entry(
                            "deepseek_v3_2",
                            new ModelSpec(
                                    "unsloth",
                                    "DeepSeek-V3.2-GGUF",
                                    "Q8_0/DeepSeek-V3.2-Q8_0-00001-of-00015.gguf")),
                    Map.entry(
                            "mistral_mistral7b_v0_3",
                            new ModelSpec(
                                    "bartowski",
                                    "Mistral-7B-Instruct-v0.3-GGUF",
                                    "Mistral-7B-Instruct-v0.3-Q8_0.gguf")));

    private static final Set<String> PARAM_MODEL_FAMILIES =
            Set.of(
                    "unsloth_llama3_2",
                    "google_gemma4",
                    "openai_gpt_oss",
                    "alibaba_qwen3_5",
                    "microsoft_phi4",
                    "ibm_granite4_1_3b",
                    "huggingface_smollm3",
                    "moonshot_kimi2_6",
                    "zai_glm5_1",
                    "minimax_m2_7",
                    "xiaomi_mimo_v2_flash",
                    "deepseek_v3_2",
                    "mistral_mistral7b_v0_3");

    // JMH @Param values are annotation literals, so keep this explicit set and fail fast if it
    // drifts from MODEL_SPECS.
    static {
        if (!new LinkedHashSet<>(MODEL_SPECS.keySet())
                .equals(new LinkedHashSet<>(PARAM_MODEL_FAMILIES))) {
            throw new IllegalStateException(
                    "GgufModelWikiBenchmark model map and @Param values drifted. map="
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
        "unsloth_llama3_2",
        "google_gemma4",
        "openai_gpt_oss",
        "alibaba_qwen3_5",
        "microsoft_phi4",
        "ibm_granite4_1_3b",
        "huggingface_smollm3",
        "moonshot_kimi2_6",
        "zai_glm5_1",
        "minimax_m2_7",
        "xiaomi_mimo_v2_flash",
        "deepseek_v3_2",
        "mistral_mistral7b_v0_3"
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
                    "Failed to initialize GGUF wiki benchmark for " + modelFamily, e);
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
                    "Unsupported GGUF model family: "
                            + familyId
                            + ". Supported: "
                            + MODEL_SPECS.keySet().stream()
                                    .sorted()
                                    .collect(Collectors.joining(", ")));
        }
        return LOADER.fromHuggingFace(spec.user(), spec.repository(), spec.ggufPath());
    }

    private record ModelSpec(String user, String repository, String ggufPath) {}
}

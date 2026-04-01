package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.FastLlama3Splitter;
import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.text.Normalizer.Form;
import java.util.Map;
import java.util.concurrent.TimeUnit;
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

/**
 * End-to-end benchmark for model tokenizer profiles (encode/decode/count), inspired by benchmark
 * comparisons used by tiktoken/HF/Mistral ecosystems.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class ModelTokenizerBenchmark {

    private static final String LLAMA3_HF_MODEL_REF = "unsloth/Llama-3.2-1B-Instruct";
    private static final String LLAMA3_HF_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c";
    private static final String QWEN35_HF_MODEL_REF = "Qwen/Qwen3-0.6B";
    private static final String MISTRAL_TEKKEN_HF_MODEL_REF = "mistralai/ministral-8b-instruct-2410";
    private static final String MISTRAL_TEKKEN_HF_REVISION =
            "2f494a194c5b980dfb9772cb92d26cbb671fce5a";

    @Param({"jtokkit", "classic", "fast"})
    public String implementation;

    @Param({"gpt2", "llama3", "qwen35", "mistral-tekken"})
    public String model;

    @Param({"chat", "code", "json", "prose", "wiki"})
    public String corpus;

    @Param({"1k", "2k", "4k", "8k", "16k", "32k"})
    public String size;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        tokenizer = createModelTokenizer(model, implementation);
        text = resize(seedText(corpus), targetLength(size));
        encoded = tokenizer.encode(text);
    }

    @Benchmark
    public void encode(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text));
    }

    @Benchmark
    public void encodeInto(Blackhole blackhole) {
        IntSequence.Builder out = IntSequence.newBuilder();
        tokenizer.encodeInto(text, out);
        blackhole.consume(out.size());
    }

    @Benchmark
    public void decode(Blackhole blackhole) {
        blackhole.consume(tokenizer.decode(encoded));
    }

    @Benchmark
    public void countTokens(Blackhole blackhole) {
        blackhole.consume(tokenizer.countTokens(text));
    }

    private static Tokenizer createModelTokenizer(String model, String implementation) {
        if ("llama3".equals(model)) {
            return modelNativeTokenizer(
                    "meta.llama3",
                    LLAMA3_HF_MODEL_REF,
                    LLAMA3_HF_REVISION,
                    Normalizer.identity(),
                    FastLlama3Splitter.INSTANCE,
                    implementation);
        }
        if ("qwen35".equals(model)) {
            return modelNativeTokenizer(
                    "alibaba.qwen3_5",
                    QWEN35_HF_MODEL_REF,
                    null,
                    Normalizer.unicode(Form.NFC),
                    FastQwen35Splitter.INSTANCE,
                    implementation);
        }
        if ("mistral-tekken".equals(model)) {
            return modelNativeTokenizer(
                    "mistral.tekken",
                    MISTRAL_TEKKEN_HF_MODEL_REF,
                    MISTRAL_TEKKEN_HF_REVISION,
                    Normalizer.identity(),
                    ModelSplitters.TEKKEN,
                    implementation);
        }
        if (!"gpt2".equals(model)) {
            throw new IllegalArgumentException("Unsupported model: " + model);
        }

        switch (implementation) {
            case "jtokkit":
                return TiktokenFixtures.createJtokkitTokenizer("r50k_base");
            case "classic":
                return classic("r50k_base", Normalizer.identity(), ModelSplitters.DEFAULT_BPE);
            case "fast":
                return fast("r50k_base", Normalizer.identity(), FastR50kSplitter.INSTANCE);
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer modelNativeTokenizer(
            String familyId,
            String hfModelRef,
            String hfRevision,
            Normalizer normalizer,
            Splitter fastSplitter,
            String implementation) {
        Tokenizer fidelityGguf =
                ModelFamilyTokenizers
                        .create(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load GGUF tokenizer for " + familyId));
        switch (implementation) {
            case "jtokkit":
                return ModelFamilyTokenizers
                        .createFromHfFiles(familyId, hfModelRef, hfRevision)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load HF tokenizer for " + familyId));
            case "classic":
                return fidelityGguf;
            case "fast":
                return Tokenizers.pipeline(fidelityGguf)
                        .normalizer(normalizer)
                        .splitter(fastSplitter)
                        .build();
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer classic(
            String encoding,
            Normalizer normalizer,
            com.qxotic.toknroll.advanced.Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return Tokenizers.classicBpe(ranks, specials, normalizer, splitter);
    }

    private static Tokenizer fast(
            String encoding,
            Normalizer normalizer,
            com.qxotic.toknroll.advanced.Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return Tokenizers.fastBpe(ranks, specials, normalizer, splitter);
    }

    private static int targetLength(String size) {
        switch (size) {
            case "1k":
                return 1024;
            case "2k":
                return 2 * 1024;
            case "4k":
                return 4 * 1024;
            case "8k":
                return 8 * 1024;
            case "16k":
                return 16 * 1024;
            case "32k":
                return 32 * 1024;
            default:
                throw new IllegalArgumentException("Unsupported size: " + size);
        }
    }

    private static String seedText(String corpus) {
        switch (corpus) {
            case "chat":
                return "<|system|>You are helpful.<|user|>Compare tokenizer throughput and"
                        + " latency.";
            case "code":
                return "for (int i = 0; i < n; i++) sum += tokenizer.countTokens(lines[i]);";
            case "json":
                return "{\"id\":123,\"items\":[{\"name\":\"alpha\",\"value\":1},"
                        + "{\"name\":\"beta\",\"value\":2}],\"ok\":true,"
                        + "\"meta\":{\"source\":\"bench\",\"tags\":[\"a\",\"b\"]}}";
            case "prose":
                return "Tokenization quality and throughput both matter for long-context systems. A"
                        + " practical benchmark should include narrative text, technical text,"
                        + " and structured data to reflect real workloads.";
            case "wiki":
                return "In computer science, tokenization is the process of converting a sequence "
                        + "of characters into a sequence of tokens, often for parsing or language "
                        + "model preprocessing.";
            default:
                throw new IllegalArgumentException("Unsupported corpus: " + corpus);
        }
    }

    private static String resize(String seed, int targetLength) {
        StringBuilder sb = new StringBuilder(targetLength + 32);
        while (sb.length() < targetLength) {
            sb.append(seed).append('\n');
        }
        if (sb.length() > targetLength) {
            sb.setLength(targetLength);
        }
        return sb.toString();
    }
}

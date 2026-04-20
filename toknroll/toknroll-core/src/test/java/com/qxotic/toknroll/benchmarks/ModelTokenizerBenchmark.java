package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.TokenizerAdapters;
import java.text.Normalizer.Form;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.AuxCounters;
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
 *
 * <p>Implementation semantics:
 *
 * <ul>
 *   <li>{@code reference}: model-matching baseline tokenizer (JTokkit for GPT-2; HF files for
 *       model-native families)
 *   <li>{@code bpe}: Tok'n'Roll model-native tokenizer
 *   <li>{@code fast}: Tok'n'Roll fast pipeline over model-native vocabulary
 * </ul>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class ModelTokenizerBenchmark {

    private static final boolean VERIFY_OUTPUTS =
            Boolean.parseBoolean(System.getProperty("toknroll.bench.verify", "true"));
    private static final ConcurrentHashMap<String, Tokenizer> REFERENCE_CACHE =
            new ConcurrentHashMap<>();

    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class DecodeCounters {
        public long decodedTokens;

        @Setup(Level.Iteration)
        public void reset() {
            decodedTokens = 0L;
        }
    }

    private static final String LLAMA3_HF_MODEL_REF = "unsloth/Llama-3.2-1B-Instruct";
    private static final String LLAMA3_HF_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c";
    private static final String QWEN35_HF_MODEL_REF = "Qwen/Qwen3-0.6B";
    private static final String GEMMA4_HF_MODEL_REF = "google/gemma-4-e2b-it";
    private static final String GPT_OSS_HF_MODEL_REF = "openai/gpt-oss-20b";
    private static final String MISTRAL_V03_HF_MODEL_REF = "mistralai/Mistral-7B-Instruct-v0.3";
    private static final String MISTRAL_TEKKEN_HF_MODEL_REF =
            "mistralai/ministral-8b-instruct-2410";
    private static final String MISTRAL_TEKKEN_HF_REVISION =
            "2f494a194c5b980dfb9772cb92d26cbb671fce5a";

    @Param({"reference", "bpe", "fast"})
    public String implementation;

    @Param({"gpt2", "gpt-oss", "llama3", "qwen35", "gemma4", "mistral-v03-spbpe", "mistral-tekken"})
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
        if (VERIFY_OUTPUTS) {
            verifyBenchmarkTokenizer(model, tokenizer, text);
        }
        encoded = tokenizer.encode(text);
    }

    private static void verifyBenchmarkTokenizer(
            String model, Tokenizer candidate, String benchmarkText) {
        Tokenizer reference =
                REFERENCE_CACHE.computeIfAbsent(
                        model, ModelTokenizerBenchmark::createReferenceTokenizer);
        for (String probe : verificationCases(model, benchmarkText)) {
            int[] expected = reference.encodeToArray(probe);
            int[] actual = candidate.encodeToArray(probe);
            if (!Arrays.equals(expected, actual)) {
                throw new IllegalStateException(
                        "Benchmark tokenizer mismatch for model="
                                + model
                                + " probe="
                                + summarizeProbe(probe)
                                + " expected="
                                + summarizeIds(expected)
                                + " actual="
                                + summarizeIds(actual));
            }
        }
    }

    private static Tokenizer createReferenceTokenizer(String model) {
        switch (model) {
            case "gpt2":
                return TiktokenFixtures.createJtokkitTokenizer("r50k_base");
            case "gpt-oss":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "openai.gpt-oss", GPT_OSS_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for openai.gpt-oss"));
            case "qwen35":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "alibaba.qwen3_5", QWEN35_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for alibaba.qwen3_5"));
            case "gemma4":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "google.gemma4", GEMMA4_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for google.gemma4"));
            case "llama3":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "meta.llama3", LLAMA3_HF_MODEL_REF, LLAMA3_HF_REVISION)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for meta.llama3"));
            case "mistral-v03-spbpe":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "mistral.v0_3_spbpe", MISTRAL_V03_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for mistral.v0_3_spbpe"));
            case "mistral-tekken":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "mistral.tekken",
                                MISTRAL_TEKKEN_HF_MODEL_REF,
                                MISTRAL_TEKKEN_HF_REVISION)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for mistral.tekken"));
            default:
                throw new IllegalArgumentException("Unsupported model: " + model);
        }
    }

    private static List<String> verificationCases(String model, String benchmarkText) {
        if ("qwen35".equals(model) || "gpt-oss".equals(model)) {
            return Arrays.asList(
                    benchmarkText,
                    "Hello, world!",
                    "<|endoftext|>",
                    "<|fim_prefix|>Hello<|fim_suffix|>World<|fim_middle|>",
                    "Hello\n\nWorld");
        }
        if ("gemma4".equals(model)) {
            return Arrays.asList(
                    benchmarkText,
                    "Hello, World!",
                    "Hello  World",
                    "Hello\n\nWorld",
                    "<html><body>Hello</body></html>");
        }
        return Arrays.asList(benchmarkText, "Hello, world!", "Hello\nWorld", "e\u0301 != é");
    }

    private static String summarizeProbe(String probe) {
        String compact = probe.replace("\n", "\\n").replace("\t", "\\t");
        return compact.length() <= 80 ? compact : compact.substring(0, 77) + "...";
    }

    private static String summarizeIds(int[] ids) {
        int max = Math.min(ids.length, 16);
        return Arrays.toString(Arrays.copyOf(ids, max)) + (ids.length > max ? "..." : "");
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
    public void decode(Blackhole blackhole, DecodeCounters counters) {
        blackhole.consume(tokenizer.decode(encoded));
        counters.decodedTokens += encoded.length();
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
                    FastSplitters.llama3(),
                    implementation);
        }
        if ("qwen35".equals(model)) {
            if ("fast".equals(implementation)) {
                return ModelFamilyTokenizers.createFast("alibaba.qwen3_5")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load fast tokenizer for"
                                                        + " alibaba.qwen3_5"));
            }
            return modelNativeTokenizer(
                    "alibaba.qwen3_5",
                    QWEN35_HF_MODEL_REF,
                    null,
                    Normalizer.unicode(Form.NFC),
                    FastSplitters.qwen35(),
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
        if ("mistral-v03-spbpe".equals(model)) {
            return modelNativeTokenizer(
                    "mistral.v0_3_spbpe",
                    MISTRAL_V03_HF_MODEL_REF,
                    null,
                    Normalizer.identity(),
                    ModelSplitters.LLAMA3,
                    implementation);
        }
        if ("gemma4".equals(model)) {
            return gemma4Tokenizer(implementation);
        }
        if ("gpt-oss".equals(model)) {
            return gptOssTokenizer(implementation);
        }
        if (!"gpt2".equals(model)) {
            throw new IllegalArgumentException("Unsupported model: " + model);
        }

        switch (implementation) {
            case "reference":
                return TiktokenFixtures.createJtokkitTokenizer("r50k_base");
            case "bpe":
                return bpe("r50k_base", Normalizer.identity(), ModelSplitters.DEFAULT_BPE);
            case "fast":
                return fast("r50k_base", Normalizer.identity(), FastSplitters.r50k());
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
                ModelFamilyTokenizers.create(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load GGUF tokenizer for " + familyId));
        switch (implementation) {
            case "reference":
                return ModelFamilyTokenizers.createFromHfFiles(familyId, hfModelRef, hfRevision)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load HF tokenizer for " + familyId));
            case "bpe":
                return fidelityGguf;
            case "fast":
                return TokenizerAdapters.withSplitter(
                        TokenizerAdapters.withNormalizer(fidelityGguf, normalizer), fastSplitter);
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer gemma4Tokenizer(String implementation) {
        Tokenizer fidelityGguf =
                ModelFamilyTokenizers.create("google.gemma4")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load GGUF tokenizer for google.gemma4"));
        if ("reference".equals(implementation)
                || "bpe".equals(implementation)
                || "fast".equals(implementation)) {
            return fidelityGguf;
        }
        throw new IllegalArgumentException("Unsupported implementation: " + implementation);
    }

    private static Tokenizer gptOssTokenizer(String implementation) {
        switch (implementation) {
            case "reference":
                return ModelFamilyTokenizers.createFromHfFiles(
                                "openai.gpt-oss", GPT_OSS_HF_MODEL_REF, null)
                        .orElseGet(
                                () ->
                                        fast(
                                                "o200k_base",
                                                Normalizer.identity(),
                                                FastSplitters.o200k()));
            case "bpe":
                return bpe("o200k_base", Normalizer.identity(), FastSplitters.o200k());
            case "fast":
                return fast("o200k_base", Normalizer.identity(), FastSplitters.o200k());
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer bpe(String encoding, Normalizer normalizer, Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return TokenizerAdapters.withNormalizer(
                TiktokenFixtures.createTikTokenTokenizer(ranks, specials, splitter), normalizer);
    }

    private static Tokenizer fast(String encoding, Normalizer normalizer, Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return TokenizerAdapters.withNormalizer(
                TiktokenFixtures.createTikTokenTokenizer(ranks, specials, splitter), normalizer);
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

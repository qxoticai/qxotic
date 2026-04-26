package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.benchmarks.support.BenchmarkSplitters;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.ByteBuffer;
import java.text.Normalizer.Form;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
 * End-to-end benchmark for model tokenizer profiles (encode/decode/count).
 *
 * <p>Implementation parameter semantics:
 *
 * <ul>
 *   <li>{@code jtokkit}: JTokkit reference implementation (GPT-2 only)
 *   <li>{@code hf-transformers}: HuggingFace Transformers reference (model-native families)
 *   <li>{@code toknroll-bpe}: Tok'n'Roll native tokenizer with default splitter
 *   <li>{@code toknroll-fast}: Tok'n'Roll with optimized fast splitter
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
    private static final String QWEN35_HF_MODEL_REF = "Qwen/Qwen3.5-0.8B";
    private static final String GEMMA4_HF_MODEL_REF = "google/gemma-4-e2b-it";
    private static final String GPT_OSS_HF_MODEL_REF = "openai/gpt-oss-20b";
    private static final String MISTRAL_V03_HF_MODEL_REF = "mistralai/Mistral-7B-Instruct-v0.3";
    private static final String MISTRAL_TEKKEN_HF_MODEL_REF =
            "mistralai/ministral-8b-instruct-2410";
    private static final String MISTRAL_TEKKEN_HF_REVISION =
            "2f494a194c5b980dfb9772cb92d26cbb671fce5a";

    @Param({"jtokkit", "toknroll-bpe", "toknroll-fast"})
    public String implementation;

    @Param({
        "gpt2",
        "openai.gpt-oss",
        "meta.llama3",
        "moonshot.kimi2_5",
        "huggingface.smollm3",
        "alibaba.qwen3_5",
        "google.gemma4",
        "nvidia.nemotron3_nano4b",
        "ibm.granite4_0",
        "microsoft.phi4",
        "deepseek.v3_2",
        "deepseek.v4_pro",
        "mistral.v0_3_spbpe",
        "mistral.tekken"
    })
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
        String canonicalModel = canonicalModelId(model);
        tokenizer = createModelTokenizer(canonicalModel, implementation);
        text = resize(seedText(corpus), targetLength(size));
        if (VERIFY_OUTPUTS) {
            verifyBenchmarkTokenizer(canonicalModel, tokenizer, text);
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
                return TestTokenizers.tiktokenReference("r50k_base");
            case "openai.gpt-oss":
                return TestTokenizers.modelFamilyFromHf(
                                "openai.gpt-oss", GPT_OSS_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for openai.gpt-oss"));
            case "alibaba.qwen3_5":
                return TestTokenizers.modelFamilyFromHf(
                                "alibaba.qwen3_5", QWEN35_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for alibaba.qwen3_5"));
            case "google.gemma4":
                return TestTokenizers.modelFamilyFromHf("google.gemma4", GEMMA4_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for google.gemma4"));
            case "moonshot.kimi2_5":
            case "huggingface.smollm3":
            case "nvidia.nemotron3_nano4b":
            case "ibm.granite4_0":
            case "microsoft.phi4":
            case "deepseek.v3_2":
            case "deepseek.v4_pro":
                return TestTokenizers.modelFamily(model)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for " + model));
            case "meta.llama3":
                return TestTokenizers.modelFamilyFromHf(
                                "meta.llama3", LLAMA3_HF_MODEL_REF, LLAMA3_HF_REVISION)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for meta.llama3"));
            case "mistral.v0_3_spbpe":
                return TestTokenizers.modelFamilyFromHf(
                                "mistral.v0_3_spbpe", MISTRAL_V03_HF_MODEL_REF, null)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "HF reference unavailable for mistral.v0_3_spbpe"));
            case "mistral.tekken":
                return TestTokenizers.modelFamilyFromHf(
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
        if ("alibaba.qwen3_5".equals(model) || "openai.gpt-oss".equals(model)) {
            return Arrays.asList(
                    benchmarkText,
                    "Hello, world!",
                    "<|endoftext|>",
                    "<|fim_prefix|>Hello<|fim_suffix|>World<|fim_middle|>",
                    "Hello\n\nWorld");
        }
        if ("google.gemma4".equals(model)) {
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
        if ("meta.llama3".equals(model)) {
            return modelNativeTokenizer(
                    "meta.llama3",
                    LLAMA3_HF_MODEL_REF,
                    LLAMA3_HF_REVISION,
                    Normalizer.identity(),
                    BenchmarkSplitters.llama3(),
                    implementation);
        }
        if ("alibaba.qwen3_5".equals(model)) {
            if ("fast".equals(implementation)) {
                return TestTokenizers.modelFamilyFast("alibaba.qwen3_5")
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
                    BenchmarkSplitters.qwen35(),
                    implementation);
        }
        if ("moonshot.kimi2_5".equals(model)
                || "huggingface.smollm3".equals(model)
                || "nvidia.nemotron3_nano4b".equals(model)
                || "ibm.granite4_0".equals(model)
                || "microsoft.phi4".equals(model)
                || "deepseek.v3_2".equals(model)
                || "deepseek.v4_pro".equals(model)) {
            return passthroughFamilyTokenizer(model, implementation);
        }
        if ("mistral.tekken".equals(model)) {
            return modelNativeTokenizer(
                    "mistral.tekken",
                    MISTRAL_TEKKEN_HF_MODEL_REF,
                    MISTRAL_TEKKEN_HF_REVISION,
                    Normalizer.identity(),
                    BenchmarkSplitters.tekken(),
                    implementation);
        }
        if ("mistral.v0_3_spbpe".equals(model)) {
            return modelNativeTokenizer(
                    "mistral.v0_3_spbpe",
                    MISTRAL_V03_HF_MODEL_REF,
                    null,
                    Normalizer.identity(),
                    BenchmarkSplitters.llama3(),
                    implementation);
        }
        if ("google.gemma4".equals(model)) {
            return gemma4Tokenizer(implementation);
        }
        if ("openai.gpt-oss".equals(model)) {
            return gptOssTokenizer(implementation);
        }
        if (!"gpt2".equals(model)) {
            throw new IllegalArgumentException("Unsupported model: " + model);
        }

        switch (implementation) {
            case "jtokkit":
                return TestTokenizers.tiktokenReference("r50k_base");
            case "toknroll-bpe":
                return bpe("r50k_base", Normalizer.identity(), BenchmarkSplitters.defaultBpe());
            case "toknroll-fast":
                return fast("r50k_base", Normalizer.identity(), BenchmarkSplitters.r50k());
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
                TestTokenizers.modelFamily(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load GGUF tokenizer for " + familyId));
        switch (implementation) {
            case "hf-transformers":
                return TestTokenizers.modelFamilyFromHf(familyId, hfModelRef, hfRevision)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load HF tokenizer for " + familyId));
            case "toknroll-bpe":
                return fidelityGguf;
            case "toknroll-fast":
                return withSplitter(withNormalizer(fidelityGguf, normalizer), fastSplitter);
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer gemma4Tokenizer(String implementation) {
        Tokenizer fidelityGguf =
                TestTokenizers.modelFamily("google.gemma4")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load GGUF tokenizer for google.gemma4"));
        if ("hf-transformers".equals(implementation)
                || "toknroll-bpe".equals(implementation)
                || "toknroll-fast".equals(implementation)) {
            return fidelityGguf;
        }
        throw new IllegalArgumentException("Unsupported implementation: " + implementation);
    }

    private static Tokenizer gptOssTokenizer(String implementation) {
        switch (implementation) {
            case "hf-transformers":
                return TestTokenizers.modelFamilyFromHf(
                                "openai.gpt-oss", GPT_OSS_HF_MODEL_REF, null)
                        .orElseGet(
                                () ->
                                        fast(
                                                "o200k_base",
                                                Normalizer.identity(),
                                                BenchmarkSplitters.o200k()));
            case "toknroll-bpe":
                return bpe("o200k_base", Normalizer.identity(), BenchmarkSplitters.o200k());
            case "toknroll-fast":
                return fast("o200k_base", Normalizer.identity(), BenchmarkSplitters.o200k());
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer passthroughFamilyTokenizer(String familyId, String implementation) {
        Tokenizer fidelity =
                TestTokenizers.modelFamily(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Failed to load tokenizer for " + familyId));
        if ("toknroll-bpe".equals(implementation) || "toknroll-fast".equals(implementation)) {
            return fidelity;
        }
        throw new IllegalArgumentException("Unsupported implementation: " + implementation);
    }

    private static Tokenizer bpe(String encoding, Normalizer normalizer, Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return withNormalizer(TestTokenizers.tiktoken(ranks, specials, splitter), normalizer);
    }

    private static Tokenizer fast(String encoding, Normalizer normalizer, Splitter splitter) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        return withNormalizer(TestTokenizers.tiktoken(ranks, specials, splitter), normalizer);
    }

    private static Tokenizer withNormalizer(Tokenizer tokenizer, Normalizer normalizer) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(normalizer, "normalizer");
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return tokenizer.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                CharSequence slice = text.subSequence(startInclusive, endExclusive);
                CharSequence transformed = normalizer.apply(slice);
                tokenizer.encodeInto(transformed, 0, transformed.length(), out);
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                CharSequence slice = text.subSequence(startInclusive, endExclusive);
                CharSequence transformed = normalizer.apply(slice);
                return tokenizer.countTokens(transformed, 0, transformed.length());
            }

            @Override
            public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
                return tokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
            }

            @Override
            public float expectedTokensPerChar() {
                return tokenizer.expectedTokensPerChar();
            }
        };
    }

    private static Tokenizer withSplitter(Tokenizer tokenizer, Splitter splitter) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(splitter, "splitter");
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return tokenizer.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                splitter.splitAll(
                        text,
                        startInclusive,
                        endExclusive,
                        (source, chunkStart, chunkEnd) ->
                                tokenizer.encodeInto(source, chunkStart, chunkEnd, out));
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                int[] total = {0};
                splitter.splitAll(
                        text,
                        startInclusive,
                        endExclusive,
                        (source, chunkStart, chunkEnd) ->
                                total[0] += tokenizer.countTokens(source, chunkStart, chunkEnd));
                return total[0];
            }

            @Override
            public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
                return tokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
            }

            @Override
            public float expectedTokensPerChar() {
                return tokenizer.expectedTokensPerChar();
            }
        };
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

    private static String canonicalModelId(String model) {
        if ("gpt-oss".equals(model)) {
            return "openai.gpt-oss";
        }
        if ("llama3".equals(model)) {
            return "meta.llama3";
        }
        if ("qwen35".equals(model)) {
            return "alibaba.qwen3_5";
        }
        if ("gemma4".equals(model)) {
            return "google.gemma4";
        }
        if ("mistral-v03-spbpe".equals(model)) {
            return "mistral.v0_3_spbpe";
        }
        if ("mistral-tekken".equals(model)) {
            return "mistral.tekken";
        }
        return model;
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

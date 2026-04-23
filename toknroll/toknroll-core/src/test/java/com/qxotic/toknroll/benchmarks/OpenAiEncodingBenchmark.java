package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.loaders.TiktokenLoaders;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.Map;
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

/** OpenAI encoding benchmark (r50k/cl100k/o200k) comparing JTokkit vs Tok'n'Roll. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class OpenAiEncodingBenchmark {

    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class DecodeCounters {
        public long decodedTokens;

        @Setup(Level.Iteration)
        public void reset() {
            decodedTokens = 0L;
        }
    }

    @Param({"jtokkit", "toknroll-bpe", "toknroll-fast"})
    public String implementation;

    @Param({"r50k_base", "cl100k_base", "o200k_base"})
    public String encoding;

    @Param({"chat", "code", "json", "prose", "wiki", "unicode"})
    public String corpus;

    @Param({"1k", "16k", "32k"})
    public String size;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        text = resize(seedText(corpus), targetLength(size));
        tokenizer = createTokenizer(encoding, implementation);
        encoded = tokenizer.encode(text);
    }

    @Benchmark
    public void encode(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text));
    }

    @Benchmark
    public void encodeInto(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text).length());
    }

    @Benchmark
    public void countTokens(Blackhole blackhole) {
        blackhole.consume(tokenizer.countTokens(text));
    }

    @Benchmark
    public void decode(Blackhole blackhole, DecodeCounters counters) {
        blackhole.consume(tokenizer.decode(encoded));
        counters.decodedTokens += encoded.length();
    }

    private static Tokenizer createTokenizer(String encoding, String implementation) {
        switch (implementation) {
            case "jtokkit":
                return TestTokenizers.tiktokenReference(encoding);
            case "toknroll-bpe":
                return createToknrollTokenizer(encoding, false);
            case "toknroll-fast":
                return createToknrollTokenizer(encoding, true);
            default:
                throw new IllegalArgumentException("Unsupported implementation: " + implementation);
        }
    }

    private static Tokenizer createToknrollTokenizer(String encoding, boolean fast) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        Splitter splitter =
                fast
                        ? fastSplitterForEncoding(encoding)
                        : Splitter.regex(TiktokenFixtures.splitPattern(encoding));
        return Tokenizers.pipeline(
                        Tokenizers.tiktokenModel(
                                TiktokenLoaders.vocabulary(ranks, specials),
                                TiktokenLoaders.mergeRules(ranks)))
                .splitter(splitter)
                .build();
    }

    private static Splitter fastSplitterForEncoding(String encoding) {
        switch (encoding) {
            case "r50k_base":
                return FastSplitters.r50k();
            case "cl100k_base":
                return FastSplitters.cl100k();
            case "o200k_base":
                return FastSplitters.o200k();
            default:
                return Splitter.regex(TiktokenFixtures.splitPattern(encoding));
        }
    }

    private static int targetLength(String size) {
        switch (size) {
            case "1k":
                return 1024;
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
                        + "{\"name\":\"beta\",\"value\":2}],\"ok\":true}";
            case "prose":
                return "Tokenization quality and throughput both matter for long-context systems. A"
                        + " practical benchmark should include narrative text, technical text,"
                        + " and structured data to reflect real workloads.";
            case "wiki":
                return "In computer science, tokenization is the process of converting a sequence"
                        + " of characters into a sequence of tokens, often for parsing or"
                        + " language model preprocessing.";
            case "unicode":
                return "你好,世界。こんにちは世界。안녕하세요 세계. Привет, мир! مرحبا بالعالم. "
                        + "नमस्ते दुनिया। Bonjour le monde! Olá mundo! Καλημέρα κόσμε. "
                        + "cafe\u0301 naive fiance\u0301 coo\u0308perate. "
                        + "Emoji test: 😀😅🤣🥲🤖🚀✨🔥🌍🧠👩‍💻👨‍👩‍👧‍👦🏳️‍🌈🇯🇵🇨🇳🇮🇳🇧🇷. "
                        + "Mixed symbols: — – • … «quotes» 『引用』 （テスト） 【测试】\n";
            default:
                throw new IllegalArgumentException("Unsupported corpus: " + corpus);
        }
    }

    private static String resize(String seed, int targetLength) {
        StringBuilder sb = new StringBuilder(targetLength + seed.length());
        while (sb.length() < targetLength) {
            sb.append(seed).append('\n');
        }
        return sb.substring(0, targetLength);
    }
}

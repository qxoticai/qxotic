package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.FastTikToken;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.charset.StandardCharsets;
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

/** Isolates FastTikToken BPE merge loop on raw bytes. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class FastTikTokenMergeBenchmark {

    @Param({"r50k_base", "cl100k_base", "o200k_base"})
    public String encoding;

    @Param({"auto", "small", "large"})
    public String path;

    @Param({"128", "512", "2048", "8192"})
    public int bytesLength;

    @Param({"chat", "code", "random", "repetitive"})
    public String corpus;

    private FastTikToken tokenizer;
    private byte[] bytes;

    @Setup(Level.Trial)
    public void setup() {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        Map<String, Integer> specials = TiktokenFixtures.specialTokens(encoding);
        tokenizer =
                (FastTikToken)
                        FastTikToken.fromTiktoken(
                                ranks, specials, Normalizer.identity(), Splitter.identity());
        bytes = resize(seedBytes(corpus), bytesLength);
    }

    @Benchmark
    public void mergeLoop(Blackhole blackhole) {
        int tokenCount;
        switch (path) {
            case "small":
                tokenCount = tokenizer.encodeBytesSmallForBenchmark(bytes);
                break;
            case "large":
                tokenCount = tokenizer.encodeBytesLargeForBenchmark(bytes);
                break;
            case "auto":
                tokenCount = tokenizer.encodeBytesForBenchmark(bytes);
                break;
            default:
                throw new IllegalArgumentException("Unsupported path: " + path);
        }
        blackhole.consume(tokenCount);
    }

    private static byte[] seedBytes(String corpus) {
        String text;
        switch (corpus) {
            case "chat":
                text =
                        "<|system|>You are helpful.<|user|>Compare tokenizer throughput and"
                                + " latency.";
                break;
            case "code":
                text = "for (int i = 0; i < n; i++) sum += tokenizer.countTokens(lines[i]);";
                break;
            case "random":
                text =
                        "Qx7!mP2@kL9#vT4$zH8%aN1^rB6&uC3*eD5(0)"
                                + "wXyZ_-=+[]{}|;:,./<>?~`"
                                + " The_quick_brown_fox_jumps_over_13_lazy_dogs.";
                break;
            case "repetitive":
                text = "aaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbcccccccccccccccccccc";
                break;
            default:
                throw new IllegalArgumentException("Unsupported corpus: " + corpus);
        }
        return text.getBytes(StandardCharsets.UTF_8);
    }

    private static byte[] resize(byte[] seed, int targetLength) {
        byte[] out = new byte[targetLength];
        int offset = 0;
        while (offset < targetLength) {
            int chunk = Math.min(seed.length, targetLength - offset);
            System.arraycopy(seed, 0, out, offset, chunk);
            offset += chunk;
        }
        return out;
    }
}

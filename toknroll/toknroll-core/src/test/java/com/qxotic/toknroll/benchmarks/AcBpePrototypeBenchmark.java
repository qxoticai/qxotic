package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.impl.AcBpeBuilder;
import com.qxotic.toknroll.impl.AcBpeModel;
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

/**
 * Prototype benchmark for AC greedy longest-match counting.
 *
 * <p>Useful for validating AC front-end speed before full backtracking integration.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class AcBpePrototypeBenchmark {

    @Param({"r50k_base", "cl100k_base", "o200k_base"})
    public String encoding;

    @Param({"4096", "8192", "16384"})
    public int maxTokens;

    @Param({"8", "16", "24"})
    public int maxTokenBytes;

    @Param({"chat", "code", "json"})
    public String corpus;

    @Param({"1k", "32k"})
    public String size;

    private AcBpeModel model;
    private byte[] bytes;

    @Setup(Level.Trial)
    public void setup() {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks(encoding);
        model = AcBpeBuilder.buildPrototype(ranks, maxTokens, maxTokenBytes);
        bytes = resize(seedText(corpus).getBytes(StandardCharsets.UTF_8), targetLength(size));
    }

    @Benchmark
    public void greedyCount(Blackhole blackhole) {
        blackhole.consume(model.greedyCount(bytes, bytes.length));
    }

    private static int targetLength(String size) {
        switch (size) {
            case "1k":
                return 1024;
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
            default:
                throw new IllegalArgumentException("Unsupported corpus: " + corpus);
        }
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

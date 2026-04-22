package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Full enwik9 benchmark for o200k fast tokenizer path. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class Enwik9FullBenchmark {

    private Tokenizer tokenizer;
    private String text;
    private volatile IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            Path path = EnwikPaths.enwik9();
            byte[] data = Files.readAllBytes(path);
            text = new String(data, StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load enwik9 corpus", e);
        }

        tokenizer = TiktokenFixtures.createTikTokenTokenizer("o200k_base", FastSplitters.o200k());
    }

    @Benchmark
    public void encode(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text));
    }

    @Benchmark
    public void decode(Blackhole blackhole) {
        IntSequence local = encoded;
        if (local == null) {
            synchronized (this) {
                local = encoded;
                if (local == null) {
                    local = tokenizer.encode(text);
                    encoded = local;
                }
            }
        }
        blackhole.consume(tokenizer.decode(local));
    }

    @Benchmark
    public void countTokens(Blackhole blackhole) {
        blackhole.consume(tokenizer.countTokens(text));
    }
}

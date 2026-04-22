package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
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

/** Full wiki corpus benchmark for o200k fast tokenizer path. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class WikiFullBenchmark {

    @Param({"enwik8", "enwik9"})
    public String corpus;

    @Param({"false", "true"})
    public boolean parallel;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            text = WikiBenchmarkSupport.loadCorpusText(corpus);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to load corpus " + corpus + " from local cache", e);
        }

        tokenizer = WikiBenchmarkSupport.createO200kTokenizer(parallel);
        encoded = tokenizer.encode(text);
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
}

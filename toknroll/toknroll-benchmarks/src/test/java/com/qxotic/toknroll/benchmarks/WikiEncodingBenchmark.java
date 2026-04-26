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

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class WikiEncodingBenchmark {

    @Param({"enwik8", "enwik9"})
    public String corpus;

    @Param({"r50k_base", "cl100k_base", "o200k_base"})
    public String encoding;

    @Param({"false", "true"})
    public boolean parallel;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            text = WikiBenchmarkSupport.loadCorpusText(corpus);
            tokenizer = WikiBenchmarkSupport.createTokenizer(encoding, parallel);
            encoded = tokenizer.encode(text);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to initialize benchmark for corpus " + corpus, e);
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
}

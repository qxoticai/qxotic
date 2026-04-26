package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.benchmarks.support.BenchmarkSplitters;
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
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Splitter-only benchmark for o200k fast splitter on full corpora. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1)
@State(Scope.Benchmark)
public class O200kSplitterOnlyBenchmark {

    @Param({"enwik8", "enwik9"})
    public String corpus;

    private String text;
    private int bytes;
    private Splitter splitter;

    @Setup(Level.Trial)
    public void setup() {
        try {
            Path path = WikiCorpusPaths.forCorpus(corpus);
            this.bytes = Math.toIntExact(Files.size(path));
            this.text = WikiBenchmarkSupport.readUtf8(path);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load benchmark corpus: " + corpus, e);
        }
        this.splitter = BenchmarkSplitters.o200k();
    }

    @Benchmark
    public void splitOnly(Blackhole blackhole) {
        final long[] chunks = {0L};
        final long[] covered = {0L};
        splitter.splitAll(
                text,
                0,
                text.length(),
                (source, start, end) -> {
                    chunks[0]++;
                    covered[0] += (end - start);
                });
        blackhole.consume(chunks[0]);
        blackhole.consume(covered[0]);
        blackhole.consume(bytes);
    }
}

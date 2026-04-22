package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.loaders.TiktokenLoaders;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
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
public class Enwik8EncodingBenchmark {

    private static final String R50K_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";
    private static final String CL100K_HASH =
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
    private static final String O200K_HASH =
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";

    @Param({"r50k_base", "cl100k_base", "o200k_base"})
    public String encoding;

    private Tokenizer tokenizer;
    private String text;
    private volatile IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            Path path = EnwikPaths.enwik8();
            byte[] data = Files.readAllBytes(path);
            text = new String(data, StandardCharsets.UTF_8);
            tokenizer = createTokenizer(encoding);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to initialize benchmark", e);
        }
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

    private static Tokenizer createTokenizer(String name) throws Exception {
        String file;
        String hash;
        Map<String, Integer> specials;
        Splitter splitter;
        if ("r50k_base".equals(name)) {
            file = "tiktoken/r50k_base.tiktoken";
            hash = R50K_HASH;
            specials = Map.of("<|endoftext|>", 50256);
            splitter = FastSplitters.r50k();
        } else if ("cl100k_base".equals(name)) {
            file = "tiktoken/cl100k_base.tiktoken";
            hash = CL100K_HASH;
            specials =
                    Map.of(
                            "<|endoftext|>", 100257,
                            "<|fim_prefix|>", 100258,
                            "<|fim_middle|>", 100259,
                            "<|fim_suffix|>", 100260,
                            "<|endofprompt|>", 100276);
            splitter = FastSplitters.cl100k();
        } else if ("o200k_base".equals(name)) {
            file = "tiktoken/o200k_base.tiktoken";
            hash = O200K_HASH;
            specials = Map.of("<|endoftext|>", 199999, "<|endofprompt|>", 200018);
            splitter = FastSplitters.o200k();
        } else {
            throw new IllegalArgumentException("Unsupported encoding: " + name);
        }

        Path resource =
                Path.of(
                        Objects.requireNonNull(
                                        Enwik8EncodingBenchmark.class
                                                .getClassLoader()
                                                .getResource(file),
                                        "Missing resource " + file)
                                .toURI());
        Map<String, Integer> ranks = TiktokenLoaders.loadMergeableRanks(resource.toString(), hash);
        TokenizationModel model =
                Tokenizers.tikTokenModel(
                        TiktokenLoaders.vocabulary(ranks, specials),
                        TiktokenLoaders.mergeRules(ranks));
        return Tokenizers.pipeline(model).splitter(splitter).build();
    }
}

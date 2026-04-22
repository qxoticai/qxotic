package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;

final class ParallelBenchmarkPipelines {
    private ParallelBenchmarkPipelines() {}

    static Tokenizer from(Tokenizer tokenizer) {
        if (!(tokenizer instanceof TokenizationPipeline)) {
            throw new IllegalStateException(
                    "Benchmark parallel wrapper requires TokenizationPipeline base instance");
        }
        TokenizationPipeline base = (TokenizationPipeline) tokenizer;
        return new ParallelTokenizationPipeline(
                base.model(),
                base.normalizer().orElse(null),
                base.splitter().orElse(null),
                ParallelTokenizationPipeline.Parallelism.fromSystemProperties());
    }
}

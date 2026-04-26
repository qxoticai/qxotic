package com.qxotic.toknroll.benchmarks;

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.CommandLineOptions;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/** Simple entrypoint to run tokenizer JMH benchmarks from IDE/CLI. */
public final class TokenizerBenchmarkRunner {

    private TokenizerBenchmarkRunner() {}

    public static void main(String[] args) throws Exception {
        if (args != null && args.length > 0) {
            new Runner(new CommandLineOptions(args)).run();
            return;
        }
        Options options =
                new OptionsBuilder()
                        .include(ModelTokenizerBenchmark.class.getSimpleName())
                        .forks(0)
                        .shouldDoGC(true)
                        .build();
        new Runner(options).run();
    }
}

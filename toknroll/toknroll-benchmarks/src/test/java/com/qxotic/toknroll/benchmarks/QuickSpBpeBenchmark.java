package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TestTokenizers;

public class QuickSpBpeBenchmark {
    public static void main(String[] args) {
        String seed = "<|system|>You are helpful.<|user|>Compare tokenizer throughput and latency.";
        StringBuilder sb = new StringBuilder(16384);
        while (sb.length() < 16384) {
            sb.append(seed).append('\n');
        }
        String text = sb.substring(0, 16384);

        benchmark("google.gemma4", text);
        benchmark("mistral.v0_3_spbpe", text);
    }

    static void benchmark(String model, String text) {
        Tokenizer tokenizer =
                TestTokenizers.modelFamily(model)
                        .orElseThrow(() -> new IllegalStateException("Failed to load " + model));

        // Warmup
        for (int i = 0; i < 50; i++) {
            tokenizer.encode(text);
        }

        // Encode benchmark
        long start = System.nanoTime();
        long ops = 0;
        while (System.nanoTime() - start < 3_000_000_000L) {
            tokenizer.encode(text);
            ops++;
        }
        long elapsedNs = System.nanoTime() - start;
        double opsPerS = ops / (elapsedNs / 1_000_000_000.0);
        double charsPerS = opsPerS * text.length();
        double mbPerS = charsPerS / 1_000_000.0;

        System.out.printf(
                "Java %s encode: %.0f ops/s, %.0f chars/s, %.2f MB/s%n",
                model, opsPerS, charsPerS, mbPerS);

        IntSequence encoded = tokenizer.encode(text);

        // Decode warmup
        for (int i = 0; i < 50; i++) {
            tokenizer.decode(encoded);
        }

        // Decode benchmark
        start = System.nanoTime();
        ops = 0;
        while (System.nanoTime() - start < 3_000_000_000L) {
            tokenizer.decode(encoded);
            ops++;
        }
        elapsedNs = System.nanoTime() - start;
        opsPerS = ops / (elapsedNs / 1_000_000_000.0);
        charsPerS = opsPerS * text.length();
        mbPerS = charsPerS / 1_000_000.0;

        System.out.printf(
                "Java %s decode: %.0f ops/s, %.0f chars/s, %.2f MB/s%n",
                model, opsPerS, charsPerS, mbPerS);
    }
}

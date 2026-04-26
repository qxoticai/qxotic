package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.benchmarks.support.BenchmarkSplitters;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.Map;

public class QuickGpt2Benchmark {
    public static void main(String[] args) {
        Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks("r50k_base");
        Map<String, Integer> specials = TiktokenFixtures.specialTokens("r50k_base");
        Tokenizer tokenizer = TestTokenizers.tiktoken(ranks, specials, BenchmarkSplitters.r50k());

        String seed = "<|system|>You are helpful.<|user|>Compare tokenizer throughput and latency.";
        StringBuilder sb = new StringBuilder(16384);
        while (sb.length() < 16384) {
            sb.append(seed).append('\n');
        }
        String text = sb.substring(0, 16384);

        // Warmup
        for (int i = 0; i < 100; i++) {
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
                "Java encode: %.0f ops/s, %.0f chars/s, %.2f MB/s%n", opsPerS, charsPerS, mbPerS);

        IntSequence encoded = tokenizer.encode(text);

        // Decode warmup
        for (int i = 0; i < 100; i++) {
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
                "Java decode: %.0f ops/s, %.0f chars/s, %.2f MB/s%n", opsPerS, charsPerS, mbPerS);
    }
}

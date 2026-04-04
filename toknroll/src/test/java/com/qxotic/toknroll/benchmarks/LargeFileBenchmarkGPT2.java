package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.impl.ClassicBPE;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Large file benchmark for testing GPT-2 (r50k_base) tokenizer performance on the enwik9 dataset.
 * This benchmark measures encoding and decoding throughput on a ~1GB file.
 */
public class LargeFileBenchmarkGPT2 {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final int ITERATIONS = 3;

    public static void main(String[] args) throws Exception {
        System.out.println("Large File Benchmark - enwik9 (GPT-2 / r50k_base)");
        System.out.println("=================================================\\n");

        // Load tokenizer
        Tokenizer tokenizer = createR50KTokenizer();

        // Read file
        Path filePath = Paths.get(FILE_PATH);
        byte[] fileBytes = Files.readAllBytes(filePath);
        String text = new String(fileBytes, java.nio.charset.StandardCharsets.UTF_8);
        long fileSizeBytes = fileBytes.length;
        double fileSizeMB = fileSizeBytes / (1024.0 * 1024.0);

        System.out.printf("File: %s%n", FILE_PATH);
        System.out.printf("Size: %.1f MB (%d bytes)%n", fileSizeMB, fileSizeBytes);
        System.out.printf("Characters: %,d%n%n", text.length());

        // Warmup
        System.out.println("Warming up...");
        IntSequence warmupTokens =
                tokenizer.encode(text.substring(0, Math.min(10000, text.length())));
        tokenizer.decode(warmupTokens);
        System.out.println("Warmup complete.\\n");

        // Run benchmark iterations
        double[] encodeTimes = new double[ITERATIONS];
        double[] decodeTimes = new double[ITERATIONS];
        int[] tokenCounts = new int[ITERATIONS];

        for (int i = 0; i < ITERATIONS; i++) {
            System.out.printf("Iteration %d/%d:%n", i + 1, ITERATIONS);

            // Encode benchmark - full file
            System.gc();
            long encodeStart = System.nanoTime();
            IntSequence tokens = tokenizer.encode(text);
            long encodeEnd = System.nanoTime();

            double encodeTime = (encodeEnd - encodeStart) / 1_000_000_000.0;
            encodeTimes[i] = encodeTime;
            tokenCounts[i] = tokens.length();

            double encodeMBps = fileSizeMB / encodeTime;
            double encodeBytesps = fileSizeBytes / encodeTime;

            System.out.printf(
                    "  Encode: %.2fs @ %.1f MB/s (%,.0f bytes/s)%n",
                    encodeTime, encodeMBps, encodeBytesps);
            System.out.printf("  Tokens: %,d%n", tokenCounts[i]);

            // Decode benchmark - sample a subset to avoid memory overflow
            int sampleTokenCount = Math.min(tokens.length(), 10000000);
            int[] allTokens = tokens.toArray();
            int[] sampleTokenArray = new int[sampleTokenCount];
            System.arraycopy(allTokens, 0, sampleTokenArray, 0, sampleTokenCount);
            IntSequence sampleTokens = IntSequence.wrap(sampleTokenArray);

            System.gc();
            long decodeStart = System.nanoTime();
            String decoded = tokenizer.decode(sampleTokens);
            long decodeEnd = System.nanoTime();

            double decodeTime = (decodeEnd - decodeStart) / 1_000_000_000.0;
            double estimatedFullDecodeTime =
                    decodeTime * ((double) tokens.length() / sampleTokenCount);
            decodeTimes[i] = estimatedFullDecodeTime;

            double decodeMBps = fileSizeMB / estimatedFullDecodeTime;
            double decodeBytesps = fileSizeBytes / estimatedFullDecodeTime;

            System.out.printf(
                    "  Decode: ~%.2fs @ %.1f MB/s (%,.0f bytes/s) [estimated]%n",
                    estimatedFullDecodeTime, decodeMBps, decodeBytesps);
            System.out.printf("    (Based on %,d tokens sample)%n%n", sampleTokenCount);
        }

        // Calculate averages
        double avgEncodeTime = average(encodeTimes);
        double avgDecodeTime = average(decodeTimes);
        double avgEncodeMBps = fileSizeMB / avgEncodeTime;
        double avgEncodeBytesps = fileSizeBytes / avgEncodeTime;
        double avgDecodeMBps = fileSizeMB / avgDecodeTime;
        double avgDecodeBytesps = fileSizeBytes / avgDecodeTime;

        System.out.println("Average Results:");
        System.out.println("================");
        System.out.printf(
                "  Encode: %.2fs @ %.1f MB/s (%,.0f bytes/s)%n",
                avgEncodeTime, avgEncodeMBps, avgEncodeBytesps);
        System.out.printf(
                "  Decode: ~%.2fs @ %.1f MB/s (%,.0f bytes/s) [estimated]%n",
                avgDecodeTime, avgDecodeMBps, avgDecodeBytesps);
        System.out.printf("  Tokens: %,d%n", tokenCounts[0]);
        System.out.printf("  Chars/Token: %.2f%n", (double) text.length() / tokenCounts[0]);
    }

    private static Tokenizer createR50KTokenizer() throws Exception {
        Path tiktokenPath =
                Path.of(
                        LargeFileBenchmarkGPT2.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks = ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        return Tokenizers.fastBpe(
                mergeableRanks, java.util.Map.of("<|endoftext|>", 50256), R50K_PATTERN);
    }

    private static double average(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }
}

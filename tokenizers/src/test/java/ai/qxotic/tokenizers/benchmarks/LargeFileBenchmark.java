package ai.qxotic.tokenizers.benchmarks;

import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.impl.ClassicBPE;
import ai.qxotic.tokenizers.impl.Tiktoken;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Pattern;

/**
 * Large file benchmark for testing tokenizer performance on the enwik9 dataset.
 * This benchmark measures encoding and decoding throughput on a ~1GB file.
 * 
 * Note: Due to Java array size limitations (Integer.MAX_VALUE), we process the file
 * in chunks for decode benchmarking.
 */
public class LargeFileBenchmark {

    private static final String CL100K_BASE_HASH =
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
    
    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+|\\s++$|\\s*[\\r\\n]|\\s+(?!\\S)|\\s";
    
    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final int ITERATIONS = 3;
    // Process in 100MB chunks to avoid memory issues
    private static final int CHUNK_SIZE = 100 * 1024 * 1024; 

    public static void main(String[] args) throws Exception {
        System.out.println("Large File Benchmark - enwik9");
        System.out.println("==============================\\n");
        
        // Load tokenizer
        Tokenizer tokenizer = createCL100KTokenizer();
        
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
        IntSequence warmupTokens = tokenizer.encode(text.substring(0, Math.min(10000, text.length())));
        tokenizer.decode(warmupTokens);
        System.out.println("Warmup complete.\\n");
        
        // Run benchmark iterations
        double[] encodeTimes = new double[ITERATIONS];
        double[] decodeTimes = new double[ITERATIONS];
        int[] tokenCounts = new int[ITERATIONS];
        
        for (int i = 0; i < ITERATIONS; i++) {
            System.out.printf("Iteration %d/%d:%n", i + 1, ITERATIONS);
            
            // Encode benchmark - full file
            System.gc(); // Suggest GC before each iteration
            long encodeStart = System.nanoTime();
            IntSequence tokens = tokenizer.encode(text);
            long encodeEnd = System.nanoTime();
            
            double encodeTime = (encodeEnd - encodeStart) / 1_000_000_000.0;
            encodeTimes[i] = encodeTime;
            tokenCounts[i] = tokens.length();
            
            double encodeMBps = fileSizeMB / encodeTime;
            double encodeBytesps = fileSizeBytes / encodeTime;
            
            System.out.printf("  Encode: %.2fs @ %.1f MB/s (%,.0f bytes/s)%n", 
                encodeTime, encodeMBps, encodeBytesps);
            System.out.printf("  Tokens: %,d%n", tokenCounts[i]);
            
            // Decode benchmark - sample a subset to avoid memory overflow
            // Take first 50MB worth of tokens for decode test
            int sampleTokenCount = Math.min(tokens.length(), 10000000); // ~10M tokens max
            int[] allTokens = tokens.toArray();
            int[] sampleTokenArray = new int[sampleTokenCount];
            System.arraycopy(allTokens, 0, sampleTokenArray, 0, sampleTokenCount);
            IntSequence sampleTokens = IntSequence.wrap(sampleTokenArray);
            
            System.gc();
            long decodeStart = System.nanoTime();
            String decoded = tokenizer.decode(sampleTokens);
            long decodeEnd = System.nanoTime();
            
            double decodeTime = (decodeEnd - decodeStart) / 1_000_000_000.0;
            // Scale up to estimate full file decode time
            double estimatedFullDecodeTime = decodeTime * ((double) tokens.length() / sampleTokenCount);
            decodeTimes[i] = estimatedFullDecodeTime;
            
            double decodeMBps = fileSizeMB / estimatedFullDecodeTime;
            double decodeBytesps = fileSizeBytes / estimatedFullDecodeTime;
            
            System.out.printf("  Decode: ~%.2fs @ %.1f MB/s (%,.0f bytes/s) [estimated]%n",
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
        System.out.printf("  Encode: %.2fs @ %.1f MB/s (%,.0f bytes/s)%n",
            avgEncodeTime, avgEncodeMBps, avgEncodeBytesps);
        System.out.printf("  Decode: ~%.2fs @ %.1f MB/s (%,.0f bytes/s) [estimated]%n",
            avgDecodeTime, avgDecodeMBps, avgDecodeBytesps);
        System.out.printf("  Tokens: %,d%n", tokenCounts[0]);
        System.out.printf("  Chars/Token: %.2f%n", (double) text.length() / tokenCounts[0]);
    }
    
    private static Tokenizer createCL100KTokenizer() throws Exception {
        Path tiktokenPath = Path.of(
                LargeFileBenchmark.class
                        .getClassLoader()
                        .getResource("tiktoken/cl100k_base.tiktoken")
                        .toURI());
        
        var mergeableRanks = ClassicBPE.loadMergeableRanks(
                tiktokenPath.toString(), CL100K_BASE_HASH);
        
        return Tiktoken.createFromTiktoken("cl100k_base", mergeableRanks,
                Pattern.compile(CL100K_PATTERN),
                java.util.Map.of(
                        "<|endoftext|>", 100257,
                        "<|fim_prefix|>", 100258,
                        "<|fim_middle|>", 100259,
                        "<|fim_suffix|>", 100260,
                        "<|endofprompt|>", 100276));
    }
    
    private static double average(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }
}

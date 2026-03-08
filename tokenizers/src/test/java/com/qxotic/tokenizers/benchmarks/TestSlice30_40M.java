package com.qxotic.tokenizers.benchmarks;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.impl.ClassicBPE;
import com.qxotic.tokenizers.impl.RegexSplitter;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Test Java tokenization on a slice (30M-40M characters) for comparison with Python. */
public class TestSlice30_40M {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final String OUTPUT_PATH =
            "/home/mukel/Desktop/playground/llm4j/tokenizers/java_tokens_slice_30_40m.txt";
    private static final int START_CHAR = 30_000_000;
    private static final int END_CHAR = 40_000_000;

    public static void main(String[] args) throws Exception {
        System.out.println("Testing Java tokenizer on slice (30M-40M chars)...");
        System.out.println("===================================================\\n");

        // Load tokenizer
        Tokenizer tokenizer = createTokenizer();

        // Read file
        Path filePath = Paths.get(FILE_PATH);
        byte[] fileBytes = Files.readAllBytes(filePath);
        String text = new String(fileBytes, java.nio.charset.StandardCharsets.UTF_8);

        // Extract slice
        String sliceText = text.substring(START_CHAR, END_CHAR);

        System.out.printf("Slice size: %,d characters%n", sliceText.length());
        System.out.println("Encoding...");

        // Encode
        IntSequence tokens = tokenizer.encode(sliceText);
        System.out.printf("Java tokens: %,d%n", tokens.length());

        // Save to file
        System.out.printf("Saving to: %s%n", OUTPUT_PATH);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_PATH))) {
            for (int i = 0; i < tokens.length(); i++) {
                writer.write(String.format("%d: %d%n", i, tokens.intAt(i)));
            }
        }

        // Show first 50 tokens
        System.out.println("\\nFirst 50 tokens:");
        for (int i = 0; i < Math.min(50, tokens.length()); i++) {
            String tokenText = tokenizer.vocabulary().token(tokens.intAt(i));
            System.out.printf("  %d: %6d -> '%s'%n", i, tokens.intAt(i), tokenText);
        }

        System.out.println("\\nDone!");
    }

    private static Tokenizer createTokenizer() throws Exception {
        Path tiktokenPath =
                Path.of(
                        TestSlice30_40M.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks = ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        return ClassicBPE.classicFromTiktoken(
                mergeableRanks,
                java.util.Map.of("<|endoftext|>", 50256),
                Normalizer.IDENTITY,
                RegexSplitter.create(R50K_PATTERN));
    }
}

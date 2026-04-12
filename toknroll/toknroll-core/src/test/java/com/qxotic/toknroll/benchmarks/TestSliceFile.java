package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.impl.ClassicBPE;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Test Java tokenization on the exact slice file from Python. */
public class TestSliceFile {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String SLICE_FILE =
            "/home/mukel/Desktop/playground/llm4j/tokenizers/slice_30_40m.txt";
    private static final String OUTPUT_PATH =
            "/home/mukel/Desktop/playground/llm4j/tokenizers/java_tokens_slice_file.txt";

    public static void main(String[] args) throws Exception {
        System.out.println("Testing Java tokenizer on slice file...");
        System.out.println("======================================\\n");

        // Load tokenizer
        Tokenizer tokenizer = createTokenizer();

        // Read slice file
        String text = Files.readString(Paths.get(SLICE_FILE));

        System.out.printf("Slice file: %s%n", SLICE_FILE);
        System.out.printf("Characters: %,d%n", text.length());
        System.out.println("Encoding...");

        // Encode
        IntSequence tokens = tokenizer.encode(text);
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
                        TestSliceFile.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks = ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        return Tokenizers.classicBpe(
                mergeableRanks, java.util.Map.of("<|endoftext|>", 50256), R50K_PATTERN);
    }
}

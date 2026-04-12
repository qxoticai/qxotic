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

/**
 * Save Java GPT2Tokenizer token sequence to file for comparison with Python. Uses the native Java
 * GPT2Tokenizer implementation instead of jtokkit.
 */
public class SaveJavaTokensGPT2 {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final String OUTPUT_PATH =
            "/home/mukel/Desktop/playground/llm4j/tokenizers/java_tokens_gpt2_native.txt";

    public static void main(String[] args) throws Exception {
        System.out.println("Saving Java GPT2Tokenizer tokens to file...");
        System.out.println("==========================================\\n");

        // Load tokenizer using native GPT2Tokenizer
        Tokenizer tokenizer = createNativeGPT2Tokenizer();

        // Read file
        Path filePath = Paths.get(FILE_PATH);
        byte[] fileBytes = Files.readAllBytes(filePath);
        String text = new String(fileBytes, java.nio.charset.StandardCharsets.UTF_8);

        System.out.printf("File: %s%n", FILE_PATH);
        System.out.printf("Characters: %,d%n", text.length());
        System.out.println("Encoding...");

        // Encode
        IntSequence tokens = tokenizer.encode(text);
        System.out.printf("Total tokens: %,d%n", tokens.length());

        // Save to file
        System.out.printf("Saving to: %s%n", OUTPUT_PATH);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_PATH))) {
            for (int i = 0; i < tokens.length(); i++) {
                writer.write(String.format("%d: %d%n", i, tokens.intAt(i)));

                if (i < 100) {
                    String tokenText = tokenizer.vocabulary().token(tokens.intAt(i));
                    System.out.printf("  %d: %6d -> '%s'%n", i, tokens.intAt(i), tokenText);
                }
            }
        }

        System.out.println("\\nDone!");
        System.out.printf("Tokens saved to: %s%n", OUTPUT_PATH);
    }

    private static Tokenizer createNativeGPT2Tokenizer() throws Exception {
        Path tiktokenPath =
                Path.of(
                        SaveJavaTokensGPT2.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks = ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        return Tokenizers.classicBpe(
                mergeableRanks, java.util.Map.of("<|endoftext|>", 50256), R50K_PATTERN);
    }
}

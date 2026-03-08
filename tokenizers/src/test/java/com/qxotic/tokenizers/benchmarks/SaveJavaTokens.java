package com.qxotic.tokenizers.benchmarks;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.impl.ClassicBPE;
import com.qxotic.tokenizers.impl.Tiktoken;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Pattern;

/** Save Java token sequence to file for comparison with Python. */
public class SaveJavaTokens {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final String OUTPUT_PATH =
            "/home/mukel/Desktop/playground/llm4j/tokenizers/java_tokens.txt";

    public static void main(String[] args) throws Exception {
        System.out.println("Saving Java tokens to file...");
        System.out.println("==============================\\n");

        // Load tokenizer
        Tokenizer tokenizer = createR50KTokenizer();

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

    private static Tokenizer createR50KTokenizer() throws Exception {
        Path tiktokenPath =
                Path.of(
                        SaveJavaTokens.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks = ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        return Tiktoken.createFromTiktoken(
                "r50k_base",
                mergeableRanks,
                Pattern.compile(R50K_PATTERN),
                java.util.Map.of("<|endoftext|>", 50256));
    }
}

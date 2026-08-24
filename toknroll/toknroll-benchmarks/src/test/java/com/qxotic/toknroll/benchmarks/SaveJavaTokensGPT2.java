package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.loaders.TiktokenLoaders;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Pattern;

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

    public static void main(String[] args) throws Exception {
        System.out.println("Saving Java GPT2Tokenizer tokens to file...");
        System.out.println("==========================================\\n");

        // Load tokenizer using native GPT2Tokenizer
        Tokenizer tokenizer = createNativeGPT2Tokenizer();

        // Read file
        Path filePath = WikiCorpusPaths.enwik9();
        byte[] fileBytes = Files.readAllBytes(filePath);
        String text = new String(fileBytes, StandardCharsets.UTF_8);

        System.out.printf("File: %s%n", filePath);
        System.out.printf("Characters: %,d%n", text.length());
        System.out.println("Encoding...");

        // Encode
        IntSequence tokens = tokenizer.encode(text);
        System.out.printf("Total tokens: %,d%n", tokens.length());

        // Save to file
        Path output = WikiCorpusPaths.benchOutput("java_tokens_gpt2_native.txt");
        System.out.printf("Saving to: %s%n", output);
        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            for (int i = 0; i < tokens.length(); i++) {
                writer.write(String.format("%d: %d%n", i, tokens.intAt(i)));

                if (i < 100) {
                    String tokenText = tokenizer.vocabulary().token(tokens.intAt(i));
                    System.out.printf("  %d: %6d -> '%s'%n", i, tokens.intAt(i), tokenText);
                }
            }
        }

        System.out.println("\\nDone!");
        System.out.printf("Tokens saved to: %s%n", output);
    }

    private static Tokenizer createNativeGPT2Tokenizer() throws Exception {
        Path tiktokenPath =
                Path.of(
                        SaveJavaTokensGPT2.class
                                .getClassLoader()
                                .getResource("tiktoken/r50k_base.tiktoken")
                                .toURI());

        var mergeableRanks =
                TiktokenLoaders.loadMergeableRanks(tiktokenPath.toString(), R50K_BASE_HASH);

        Vocabulary vocabulary =
                TiktokenLoaders.vocabulary(mergeableRanks, Map.of("<|endoftext|>", 50256));
        return Toknroll.pipeline(
                Splitter.regex(Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)),
                Toknroll.tiktokenModel(vocabulary, TiktokenLoaders.mergeRules(mergeableRanks)));
    }
}

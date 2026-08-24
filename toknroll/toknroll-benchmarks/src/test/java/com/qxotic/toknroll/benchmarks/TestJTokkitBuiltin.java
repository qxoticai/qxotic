package com.qxotic.toknroll.benchmarks;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;

/** Test jtokkit's built-in R50K_BASE tokenizer. */
public class TestJTokkitBuiltin {

    public static void main(String[] args) throws Exception {
        System.out.println("Testing jtokkit built-in R50K_BASE tokenizer...");
        System.out.println("=============================================\\n");

        // Load jtokkit's built-in R50K_BASE tokenizer
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        Encoding encoding = registry.getEncoding(EncodingType.R50K_BASE);

        // Read slice file
        Path sliceFile = WikiCorpusPaths.slice30To40M();
        String text = Files.readString(sliceFile);

        System.out.printf("Slice file: %s%n", sliceFile);
        System.out.printf("Characters: %,d%n", text.length());
        System.out.println("Encoding...");

        // Encode
        IntArrayList tokens = encoding.encode(text);
        System.out.printf("jtokkit tokens: %,d%n", tokens.size());

        // Save to file
        Path output = WikiCorpusPaths.benchOutput("java_tokens_jtokkit_builtin.txt");
        System.out.printf("Saving to: %s%n", output);
        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            for (int i = 0; i < tokens.size(); i++) {
                writer.write(String.format("%d: %d%n", i, tokens.get(i)));
            }
        }

        // Show first 50 tokens
        System.out.println("\\nFirst 50 tokens:");
        for (int i = 0; i < Math.min(50, tokens.size()); i++) {
            int tokenId = tokens.get(i);
            IntArrayList singleToken = new IntArrayList(1);
            singleToken.add(tokenId);
            String tokenText = encoding.decode(singleToken);
            System.out.printf("  %d: %6d -> '%s'%n", i, tokenId, tokenText);
        }

        System.out.println("\\nDone!");
    }
}

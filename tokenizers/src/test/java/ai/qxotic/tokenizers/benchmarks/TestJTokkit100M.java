package ai.qxotic.tokenizers.benchmarks;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Test jtokkit's built-in R50K_BASE tokenizer on 100M character slice.
 */
public class TestJTokkit100M {

    private static final String FILE_PATH = "/home/mukel/Desktop/playground/enwik9";
    private static final String OUTPUT_PATH = "/home/mukel/Desktop/playground/llm4j/tokenizers/java_tokens_100m_slice.txt";
    private static final int START_CHAR = 30_000_000;
    private static final int END_CHAR = 130_000_000;  // 100M characters

    public static void main(String[] args) throws Exception {
        System.out.println("Testing jtokkit built-in R50K_BASE on 100M character slice...");
        System.out.println("=========================================================\\n");
        
        // Load jtokkit's built-in R50K_BASE tokenizer
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        Encoding encoding = registry.getEncoding(EncodingType.R50K_BASE);
        
        // Read file and extract slice
        System.out.println("Reading file and extracting 100M character slice...");
        String text = Files.readString(Paths.get(FILE_PATH));
        String sliceText = text.substring(START_CHAR, END_CHAR);
        
        System.out.printf("Slice: characters %,d to %,d%n", START_CHAR, END_CHAR);
        System.out.printf("Slice size: %,d characters%n", sliceText.length());
        System.out.println("Encoding...");
        
        // Encode
        long startTime = System.currentTimeMillis();
        IntArrayList tokens = encoding.encode(sliceText);
        long endTime = System.currentTimeMillis();
        
        double seconds = (endTime - startTime) / 1000.0;
        double mbps = (sliceText.length() / (1024.0 * 1024.0)) / seconds;
        
        System.out.printf("jtokkit tokens: %,d%n", tokens.size());
        System.out.printf("Time: %.2f seconds (%.1f MB/s)%n", seconds, mbps);
        
        // Save to file
        System.out.printf("Saving to: %s%n", OUTPUT_PATH);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_PATH))) {
            for (int i = 0; i < tokens.size(); i++) {
                writer.write(String.format("%d: %d%n", i, tokens.get(i)));
            }
        }
        
        System.out.println("\\nDone!");
    }
}

package com.qxotic.tokenizers.hf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizersLocalTest {

    @TempDir Path tempDir;

    @Test
    void loadFromTokenizerJson() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(tokenizerJson, minimalTokenizerJson());

        Tokenizer tokenizer = HuggingFaceTokenizers.fromTokenizerJson(tokenizerJson);
        assertNotNull(tokenizer);
        assertEquals(3, tokenizer.vocabulary().size());

        IntSequence tokens = tokenizer.encode("Hi");
        assertNotNull(tokens);
    }

    @Test
    void loadFromDirectory() throws IOException {
        Files.writeString(tempDir.resolve("tokenizer.json"), minimalTokenizerJson());
        Tokenizer tokenizer = HuggingFaceTokenizers.fromDirectory(tempDir);
        assertNotNull(tokenizer);
    }

    @Test
    void loadFromDirectoryFailsWhenTokenizerJsonMissing() {
        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> HuggingFaceTokenizers.fromDirectory(tempDir));
        assertNotNull(ex.getMessage());
    }

    @Test
    void loadFromFilesUsesExternalMerges() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Path merges = tempDir.resolve("merges.txt");

        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"H": 0, "i": 1, "Hi": 2}
                  },
                  "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": "."}
                  }
                }
                """);
        Files.writeString(merges, "H i\n");

        Tokenizer tokenizer = HuggingFaceTokenizers.fromFiles(tokenizerJson, merges, null);
        assertNotNull(tokenizer);
        assertEquals(3, tokenizer.vocabulary().size());
    }

    private static String minimalTokenizerJson() {
        return """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"H": 0, "i": 1, "Hi": 2},
                    "merges": [["H", "i"]]
                  },
                  "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": "."}
                  }
                }
                """;
    }
}

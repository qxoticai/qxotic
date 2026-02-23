package com.qxotic.tokenizers.hf.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HfTokenizerJsonParserTest {

    @TempDir Path tempDir;

    @Test
    void parseBpeTokenizerWithInlineMergesAndRegex() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"H": 0, "i": 1, "Hi": 2},
                    "merges": [["H", "i"]]
                  },
                  "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                      {"type": "Split", "pattern": {"Regex": "."}},
                      {"type": "ByteLevel"}
                    ]
                  },
                  "added_tokens": [{"id": 100, "content": "<|eot|>"}]
                }
                """);

        HfTokenizerJsonParser parser = new HfTokenizerJsonParser(tokenizerJson);
        HfTokenizerJsonParser.TokenizerData data = parser.parse();

        assertEquals(".", data.regex());
        assertEquals(3, data.vocab().size());
        assertEquals(1, data.merges().size());
        assertEquals("H i", data.merges().get(0));
        assertEquals(100, data.addedTokens().get("<|eot|>"));
    }

    @Test
    void parseLoadsMergesFromExternalFileWhenMissingInline() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Path mergesTxt = tempDir.resolve("merges.txt");

        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2}
                  },
                  "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": "."}
                  }
                }
                """);
        Files.writeString(mergesTxt, "#version: 0.2\na b\n");

        HfTokenizerJsonParser parser = new HfTokenizerJsonParser(tokenizerJson, mergesTxt);
        HfTokenizerJsonParser.TokenizerData data = parser.parse();

        assertEquals(1, data.merges().size());
        assertEquals("a b", data.merges().get(0));
    }

    @Test
    void parseFailsForUnsupportedModelType() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {"type": "WordPiece", "vocab": {}},
                  "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}}
                }
                """);

        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> new HfTokenizerJsonParser(tokenizerJson).parse());

        assertTrue(ex.getMessage().contains("Only BPE is supported"));
    }

    @Test
    void parseFailsWhenRegexMissing() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"a": 0},
                    "merges": []
                  },
                  "pre_tokenizer": {"type": "ByteLevel"}
                }
                """);

        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> new HfTokenizerJsonParser(tokenizerJson).parse());

        assertTrue(ex.getMessage().contains("No explicit regex pattern"));
    }

    @Test
    void parseFailsOnDuplicateTokenIds() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(
                tokenizerJson,
                """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {"a": 1, "b": 1},
                    "merges": []
                  },
                  "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}}
                }
                """);

        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> new HfTokenizerJsonParser(tokenizerJson).parse());

        assertTrue(ex.getMessage().contains("Duplicate token ID"));
    }
}

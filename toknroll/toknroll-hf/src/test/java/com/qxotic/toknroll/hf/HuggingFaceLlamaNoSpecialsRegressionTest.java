package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class HuggingFaceLlamaNoSpecialsRegressionTest {

    private static final Path CORE_GOLDEN_ENWIK8_DIR =
            Path.of("..", "toknroll-core", "src", "test", "resources", "golden", "enwik8");

    @Test
    @SuppressWarnings("unchecked")
    void llamaParityAgainstEnwik8FixtureFirstChunk() throws IOException {
        Path chunksPath =
                CORE_GOLDEN_ENWIK8_DIR.resolve("chunks.json").toAbsolutePath().normalize();
        Path groundTruthPath =
                CORE_GOLDEN_ENWIK8_DIR
                        .resolve("hf_unsloth_llama3_2_ground_truth.json")
                        .toAbsolutePath()
                        .normalize();

        assertTrue(Files.exists(chunksPath), "Missing chunks fixture: " + chunksPath);
        assertTrue(Files.exists(groundTruthPath), "Missing llama fixture: " + groundTruthPath);

        List<Object> chunkRows =
                (List<Object>) Json.parse(Files.readString(chunksPath, StandardCharsets.UTF_8));
        List<Object> gtRows =
                (List<Object>)
                        Json.parse(Files.readString(groundTruthPath, StandardCharsets.UTF_8));

        Map<String, Object> firstChunk = (Map<String, Object>) chunkRows.get(0);
        Map<String, Object> firstGt = (Map<String, Object>) gtRows.get(0);

        String chunkHash = (String) firstChunk.get("hash");
        String expectedHash = (String) firstGt.get("chunk_hash");
        assertEquals(expectedHash, chunkHash, "Fixture ordering mismatch for first enwik8 chunk");

        String text = (String) firstChunk.get("text");
        List<Object> tokenList = (List<Object>) firstGt.get("tokens");
        int[] expected = new int[tokenList.size()];
        for (int i = 0; i < tokenList.size(); i++) {
            expected[i] = ((Number) tokenList.get(i)).intValue();
        }

        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.fromPretrained(
                        "unsloth", "Llama-3.2-1B-Instruct", "main", false, false);
        int[] actual = tokenizer.encodeToArray(text);

        assertArrayEquals(expected, actual, "Llama fixture drift detected for first enwik8 chunk");
        assertEquals(expected.length, tokenizer.countTokens(text));
        assertEquals(text, tokenizer.decode(IntSequence.wrap(expected)));
    }
}

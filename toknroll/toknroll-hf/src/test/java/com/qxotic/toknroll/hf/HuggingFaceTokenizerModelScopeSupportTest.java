package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerModelScopeSupportTest {

    private static final String CACHE_ROOT_PROPERTY = "toknroll.cache.root";

    @TempDir Path tempDir;

    @Test
    void fromModelScope_defaultsToMasterRevision() throws IOException {
        writeModelScopeCacheFile(
                "org",
                "repo",
                "master",
                "tokenizer.json",
                "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}");

        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromModelScope("org", "repo");
            assertEquals(1, tokenizer.vocabulary().size());
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    @Test
    void fromModelScope_blankRevisionRejected() {
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                HuggingFaceTokenizerLoader.fromModelScope(
                                        "org", "repo", "   ", true, false));
        assertTrue(error.getMessage().contains("revision"));
    }

    @Test
    void loadFromModelScopeTiktokenModel_buildsTokenizer() throws Exception {
        writeModelScopeCacheFile(
                "org",
                "repo",
                "dev",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("xy", 256)));
        writeModelScopeCacheFile(
                "org",
                "repo",
                "dev",
                "tokenizer_config.json",
                "{\"pat_str\":\"[a-z]+\",\"added_tokens_decoder\":{\"99\":{\"content\":\"<|end|>\"}}}");

        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.loadFromModelScopeTiktokenModel(
                        RepositoryArtifactCache.create(tempDir), "org", "repo", "dev", true, false);

        assertArrayEquals(new int[] {256}, tokenizer.encode("xy").toArray());
        assertTrue(tokenizer.vocabulary().contains("<|end|>"));
    }

    @Test
    void fromModelScope_cacheOnlyCanFallbackToCachedTiktokenModel() throws Exception {
        writeModelScopeCacheFile(
                "cacheonly",
                "demo",
                "master",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("xy", 256)));
        writeModelScopeCacheFile(
                "cacheonly", "demo", "master", "tokenizer_config.json", "{\"pat_str\":\"[a-z]+\"}");

        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            Tokenizer tokenizer =
                    HuggingFaceTokenizerLoader.fromModelScope(
                            "cacheonly", "demo", null, true, false);
            assertArrayEquals(new int[] {256}, tokenizer.encode("xy").toArray());
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    private Path writeModelScopeCacheFile(
            String user, String repository, String revision, String fileName, String content)
            throws IOException {
        Path target =
                tempDir.resolve("repository-artifacts")
                        .resolve("modelscope")
                        .resolve(user)
                        .resolve(repository)
                        .resolve(revision)
                        .resolve(fileName);
        Files.createDirectories(target.getParent());
        return Files.writeString(target, content, StandardCharsets.UTF_8);
    }

    private static String tiktokenWithBaseBytes(String... extras) {
        String[] lines = new String[256 + extras.length];
        for (int b = 0; b < 256; b++) {
            lines[b] = tokenLine(new byte[] {(byte) b}, b);
        }
        for (int i = 0; i < extras.length; i++) {
            lines[256 + i] = extras[i];
        }
        return tiktokenLines(lines);
    }

    private static String tiktokenLines(String... lines) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < lines.length; i++) {
            if (i > 0) {
                sb.append('\n');
            }
            sb.append(lines[i]);
        }
        sb.append('\n');
        return sb.toString();
    }

    private static String tokenLine(String token, int rank) {
        String base64 = Base64.getEncoder().encodeToString(token.getBytes(StandardCharsets.UTF_8));
        return base64 + " " + rank;
    }

    private static String tokenLine(byte[] tokenBytes, int rank) {
        String base64 = Base64.getEncoder().encodeToString(tokenBytes);
        return base64 + " " + rank;
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}

package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.testkit.TestSystemProperties;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerLoadExceptionTest {
    private static final String CACHE_ROOT_PROPERTY = TestSystemProperties.ARTIFACT_CACHE_ROOT;

    @TempDir Path tempDir;

    @Test
    void fromHuggingFace_cacheOnlyMissThrowsTokenizerLoadException() {
        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            TokenizerLoadException error =
                    assertThrows(
                            TokenizerLoadException.class,
                            () ->
                                    HuggingFaceTokenizerLoader.fromHuggingFace(
                                            "missing-user", "missing-repo", "main", true, false));
            assertTrue(error.getMessage().contains("[huggingface]"));
            assertTrue(error.getCause() instanceof IOException);
            assertTrue(error.getCause().getMessage().contains("useCacheOnly=true"));
            assertTrue(error.getCause().getMessage().contains("[huggingface]"));
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    @Test
    void fromModelScope_cacheOnlyMissThrowsTokenizerLoadException() {
        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            TokenizerLoadException error =
                    assertThrows(
                            TokenizerLoadException.class,
                            () ->
                                    HuggingFaceTokenizerLoader.fromModelScope(
                                            "missing-user", "missing-repo", null, true, false));
            assertTrue(error.getMessage().contains("[modelscope]"));
            assertTrue(error.getCause() instanceof IOException);
            assertTrue(error.getCause().getMessage().contains("useCacheOnly=true"));
            assertTrue(error.getCause().getMessage().contains("[modelscope]"));
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    @Test
    void fromHuggingFace_invalidCachedTokenizerJsonWrapsAsTokenizerLoadException()
            throws Exception {
        writeCachedHuggingFaceFile("alice", "broken", "main", "tokenizer.json", "not-json");
        writeCachedHuggingFaceFile("alice", "broken", "main", "tokenizer_config.json", "{}");
        writeCachedHuggingFaceFile("alice", "broken", "main", "special_tokens_map.json", "{}");
        writeCachedHuggingFaceFile("alice", "broken", "main", "added_tokens.json", "{}");

        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            RuntimeException error =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    HuggingFaceTokenizerLoader.fromHuggingFace(
                                            "alice", "broken", "main", true, false));
            assertTrue(error.getMessage() != null && !error.getMessage().isBlank());
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    private Path writeCachedHuggingFaceFile(
            String user, String repository, String revision, String fileName, String content)
            throws IOException {
        Path target =
                tempDir.resolve("repository-artifacts")
                        .resolve("huggingface")
                        .resolve(user)
                        .resolve(repository)
                        .resolve(revision)
                        .resolve(fileName);
        Files.createDirectories(target.getParent());
        return Files.writeString(target, content, StandardCharsets.UTF_8);
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}

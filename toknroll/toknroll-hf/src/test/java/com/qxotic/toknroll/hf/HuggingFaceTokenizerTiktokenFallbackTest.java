package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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

class HuggingFaceTokenizerTiktokenFallbackTest {

    private static final String CACHE_ROOT_PROPERTY = "toknroll.cache.root";

    @TempDir Path tempDir;

    @Test
    void tiktokenFallbackLoadsSpecialTokensFromTokenizerConfig() throws Exception {
        writeHfCacheFile(
                "alice",
                "demo",
                "main",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("ab", 256), tokenLine("cd", 257)));
        writeHfCacheFile(
                "alice",
                "demo",
                "main",
                "tokenizer_config.json",
                "{\"pat_str\":\"[a-z]+|\\\\s+\",\"added_tokens_decoder\":{\"99\":{\"content\":\"<|end|>\"}}}");

        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.loadFromHfTiktokenModel(
                        RepositoryArtifactCache.create(tempDir),
                        "alice",
                        "demo",
                        "main",
                        true,
                        false);

        assertArrayEquals(new int[] {256, 32, 257}, tokenizer.encode("ab cd").toArray());
        assertTrue(tokenizer.vocabulary().id("<|end|>") >= 258);
    }

    @Test
    void tiktokenFallbackParsesPatStrFromAutoMapModule() throws Exception {
        writeHfCacheFile(
                "bob",
                "demo",
                "main",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("xy", 256)));
        writeHfCacheFile(
                "bob",
                "demo",
                "main",
                "tokenizer_config.json",
                "{\"auto_map\":{\"AutoTokenizer\":[\"tokenization_demo.CustomTokenizer\"]}}" + "");
        writeHfCacheFile(
                "bob",
                "demo",
                "main",
                "tokenization_demo.py",
                "pat_str = \"|\".join([\n"
                        + "r\"\"\"\\p{Han}+\"\"\",\n"
                        + "r\"\"\"xy\"\"\"\n"
                        + "])\n");

        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.loadFromHfTiktokenModel(
                        RepositoryArtifactCache.create(tempDir),
                        "bob",
                        "demo",
                        "main",
                        true,
                        false);

        assertArrayEquals(new int[] {256}, tokenizer.encode("xy").toArray());
    }

    @Test
    void parsePatStrFromPythonModuleRewritesHanClassForJavaRegex() {
        String source =
                "pat_str = \"|\".join([\n"
                        + "r\"\"\"\\p{Han}+\"\"\",\n"
                        + "r\"\"\"[a-z]+\"\"\"\n"
                        + "])\n";
        String pattern = HuggingFaceTokenizerLoader.parsePatStrFromPythonModule(source);

        assertTrue(pattern.contains("\\p{IsHan}"));
        assertTrue(pattern.contains("[a-z]+"));
    }

    @Test
    void tiktokenFallbackFailsFastWhenPatStrIsUnavailable() throws Exception {
        writeHfCacheFile(
                "nopat",
                "demo",
                "main",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("xy", 256)));
        writeHfCacheFile("nopat", "demo", "main", "tokenizer_config.json", "{}");

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                HuggingFaceTokenizerLoader.loadFromHfTiktokenModel(
                                        RepositoryArtifactCache.create(tempDir),
                                        "nopat",
                                        "demo",
                                        "main",
                                        true,
                                        false));
        assertTrue(error.getMessage().contains("requires pat_str"));
    }

    @Test
    void fromHuggingFace_cacheOnlyCanFallbackToCachedTiktokenModel() throws Exception {
        writeHfCacheFile(
                "cacheonly",
                "demo",
                "main",
                "tiktoken.model",
                tiktokenWithBaseBytes(tokenLine("xy", 256)));
        writeHfCacheFile(
                "cacheonly", "demo", "main", "tokenizer_config.json", "{\"pat_str\":\"[a-z]+\"}");

        String previous = System.getProperty(CACHE_ROOT_PROPERTY);
        System.setProperty(CACHE_ROOT_PROPERTY, tempDir.toString());
        try {
            Tokenizer tokenizer =
                    HuggingFaceTokenizerLoader.fromHuggingFace(
                            "cacheonly", "demo", "main", true, false);
            assertArrayEquals(new int[] {256}, tokenizer.encode("xy").toArray());
        } finally {
            restoreProperty(CACHE_ROOT_PROPERTY, previous);
        }
    }

    private Path writeHfCacheFile(
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

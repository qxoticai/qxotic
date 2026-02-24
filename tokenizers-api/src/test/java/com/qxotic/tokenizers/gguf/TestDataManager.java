package com.qxotic.tokenizers.gguf;

import com.qxotic.format.gguf.GGUF;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages downloading and caching of GGUF model metadata from Hugging Face.
 *
 * <p>This class downloads only the metadata portion of GGUF files (not the full model weights) and
 * caches them locally for testing purposes. The cache is stored in {@code
 * ~/.cache/qxotic-tokenizers/gguf-metadata/}.
 *
 * <p>To invalidate the cache, simply delete the cache directory.
 */
public class TestDataManager {

    private static final String CACHE_DIR =
            System.getProperty("user.home") + "/.cache/qxotic/tokenizers/gguf-metadata";

    private final HttpClient httpClient;
    private final Path cachePath;
    private final Map<String, GGUF> loadedMetadata;

    /** Predefined models for testing (publicly accessible, no auth required). */
    public enum TestModel {
        // Qwen models - excellent quality, permissive license, publicly accessible
        QWEN3_0_6B("Qwen", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        QWEN2_5_0_5B("Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),

        // Gemma 3 from unsloth (potentially publicly accessible)
        GEMMA_3_4B_UNSLOTH("unsloth", "gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),

        // Ministral from bartowski (potentially publicly accessible)
        MISTRAL_3_3B_BARTOWSKI("bartowski", "mistralai_Ministral-3-3B-Instruct-2512-GGUF", "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    ;

        private final String org;
        private final String repo;
        private final String filename;

        TestModel(String org, String repo, String filename) {
            this.org = org;
            this.repo = repo;
            this.filename = filename;
        }

        public String getOrg() {
            return org;
        }

        public String getRepo() {
            return repo;
        }

        public String getFilename() {
            return filename;
        }

        public String getHuggingFaceUrl() {
            return String.format(
                    "https://huggingface.co/%s/%s/resolve/main/%s", org, repo, filename);
        }

        public String getCacheKey() {
            return String.format("%s_%s_%s", org, repo, filename.replace(".gguf", ""));
        }
    }

    public TestDataManager() {
        this(Paths.get(CACHE_DIR));
    }

    public TestDataManager(Path cachePath) {
        this.httpClient =
                HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        this.cachePath = cachePath;
        this.loadedMetadata = new ConcurrentHashMap<>();
        ensureCacheDirectory();
    }

    /**
     * Gets or downloads the GGUF metadata for a test model.
     *
     * @param model the test model to load
     * @return the GGUF metadata
     * @throws IOException if download or parsing fails
     * @throws InterruptedException if download is interrupted
     */
    public GGUF getOrDownloadMetadata(TestModel model) throws IOException, InterruptedException {
        String cacheKey = model.getCacheKey();

        // Check if already loaded in memory
        GGUF cached = loadedMetadata.get(cacheKey);
        if (cached != null) {
            return cached;
        }

        // Check if cached on disk
        Path cachedFile = cachePath.resolve(cacheKey + ".gguf.partial");
        if (Files.exists(cachedFile)) {
            GGUF gguf = loadFromCache(cachedFile);
            loadedMetadata.put(cacheKey, gguf);
            return gguf;
        }

        // Download from Hugging Face
        downloadAndCache(model, cachedFile);

        GGUF gguf = loadFromCache(cachedFile);
        loadedMetadata.put(cacheKey, gguf);
        return gguf;
    }

    /**
     * Downloads only the metadata portion of a GGUF file. Uses HTTP range requests to download just
     * the header and metadata.
     */
    private void downloadAndCache(TestModel model, Path targetFile)
            throws IOException, InterruptedException {
        String url = model.getHuggingFaceUrl();

        // Try downloading progressively larger chunks until we can parse the metadata
        int[] sizesToTry = {2, 5, 10, 20}; // MB
        byte[] data = null;

        for (int sizeMB : sizesToTry) {
            int sizeBytes = sizeMB * 1024 * 1024;

            HttpRequest request =
                    HttpRequest.newBuilder()
                            .uri(URI.create(url))
                            .header("Range", "bytes=0-" + (sizeBytes - 1))
                            .GET()
                            .build();

            HttpResponse<byte[]> response =
                    httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());

            if (response.statusCode() != 200 && response.statusCode() != 206) {
                throw new IOException("Failed to download: HTTP " + response.statusCode());
            }

            data = response.body();

            // Verify it's a valid GGUF file by checking magic number
            if (data.length < 4
                    || data[0] != 'G'
                    || data[1] != 'G'
                    || data[2] != 'U'
                    || data[3] != 'F') {
                throw new IOException(
                        "Downloaded file is not a valid GGUF file (wrong magic number)");
            }

            // Try to parse it to see if we have enough data
            try {
                Path tempFile = Files.createTempFile("gguf-test-", ".partial");
                Files.write(tempFile, data);
                GGUF gguf = GGUF.read(tempFile);
                // If we get here, we have enough data
                Files.deleteIfExists(tempFile);
                break;
            } catch (Exception e) {
                if (sizeMB == sizesToTry[sizesToTry.length - 1]) {
                    throw new IOException(
                            "Failed to download enough metadata after trying " + sizeMB + "MB", e);
                }
            }
        }

        // Save to cache
        Files.createDirectories(targetFile.getParent());
        Files.write(
                targetFile, data, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    /** Loads GGUF metadata from a cached partial file. */
    private GGUF loadFromCache(Path cachedFile) throws IOException {
        try (ReadableByteChannel channel =
                Files.newByteChannel(cachedFile, StandardOpenOption.READ)) {
            return GGUF.read(channel);
        }
    }

    /** Clears all cached metadata files. */
    public void clearCache() throws IOException {
        if (Files.exists(cachePath)) {
            Files.walk(cachePath)
                    .sorted((a, b) -> -a.compareTo(b)) // Reverse order to delete files before dirs
                    .forEach(
                            path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException e) {
                                    System.err.println("Failed to delete: " + path);
                                }
                            });
        }
        loadedMetadata.clear();
        ensureCacheDirectory();
    }

    /** Returns the path to the cache directory. */
    public Path getCachePath() {
        return cachePath;
    }

    private void ensureCacheDirectory() {
        try {
            Files.createDirectories(cachePath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create cache directory: " + cachePath, e);
        }
    }

    /**
     * Extracts tokenizer vocabulary from GGUF metadata.
     *
     * @param gguf the GGUF metadata
     * @return TokenizerMetadata containing vocabulary and related info
     */
    public static TokenizerMetadata extractTokenizerMetadata(GGUF gguf) {
        String modelType = gguf.getValueOrDefault(String.class, "tokenizer.ggml.model", "unknown");
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        float[] scores = gguf.getValueOrDefault(float[].class, "tokenizer.ggml.scores", null);
        int[] tokenTypes = gguf.getValueOrDefault(int[].class, "tokenizer.ggml.token_type", null);
        String[] merges = gguf.getValueOrDefault(String[].class, "tokenizer.ggml.merges", null);

        Integer bosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", null);
        Integer eosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", null);
        Integer padTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.pad_token_id", null);
        Integer unkTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.unk_token_id", null);

        String modelName = gguf.getValueOrDefault(String.class, "general.name", "unknown");
        String architecture =
                gguf.getValueOrDefault(String.class, "general.architecture", "unknown");

        return new TokenizerMetadata(
                modelType,
                modelName,
                architecture,
                tokens,
                scores,
                tokenTypes,
                merges,
                bosTokenId,
                eosTokenId,
                padTokenId,
                unkTokenId);
    }

    /** Holder for tokenizer metadata extracted from GGUF. */
    public record TokenizerMetadata(
            String modelType,
            String modelName,
            String architecture,
            String[] tokens,
            float[] scores,
            int[] tokenTypes,
            String[] merges,
            Integer bosTokenId,
            Integer eosTokenId,
            Integer padTokenId,
            Integer unkTokenId) {
        public int vocabularySize() {
            return tokens != null ? tokens.length : 0;
        }

        public boolean isBpe() {
            return "gpt2".equalsIgnoreCase(modelType) || "bpe".equalsIgnoreCase(modelType);
        }

        public boolean isSentencePiece() {
            return "llama".equalsIgnoreCase(modelType)
                    || "sp".equalsIgnoreCase(modelType)
                    || "sentencepiece".equalsIgnoreCase(modelType);
        }
    }
}

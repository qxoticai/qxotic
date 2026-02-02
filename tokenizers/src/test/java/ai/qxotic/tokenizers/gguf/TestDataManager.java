package ai.qxotic.tokenizers.gguf;

import ai.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
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
 * <p>This class downloads only the metadata portion of GGUF files (not the full model weights)
 * and caches them locally for testing purposes. The cache is stored in 
 * {@code ~/.cache/qxotic-tokenizers/gguf-metadata/}.
 * 
 * <p>To invalidate the cache, simply delete the cache directory.
 */
public class TestDataManager {
    
    private static final String CACHE_DIR = System.getProperty("user.home") + 
            "/.cache/qxotic-tokenizers/gguf-metadata";
    private static final int DOWNLOAD_BUFFER_SIZE = 8192;
    private static final int MAX_METADATA_SIZE = 10 * 1024 * 1024; // 10MB max for metadata
    
    private final HttpClient httpClient;
    private final Path cachePath;
    private final Map<String, GGUF> loadedMetadata;
    
    /**
     * Predefined small models for testing.
     */
    public enum TestModel {
        GEMMA_3_1B("bartowski", "google_gemma-3-1b-it-GGUF", "google_gemma-3-1b-it-Q4_K_M.gguf"),
        QWEN3_0_6B("Qwen", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        MISTRAL_SMALL_3_1("bartowski", "mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF", 
                "mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf"),
        // Note: Llama 3.2 requires Hugging Face authentication
        // LLAMA_3_2_1B("bartowski", "meta-llama_Llama-3.2-1B-Instruct-GGUF", 
        //         "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        // Note: Phi-4 requires Hugging Face authentication  
        // PHI_4_MINI("bartowski", "microsoft_Phi-4-mini-instruct-GGUF",
        //         "Phi-4-Mini-Instruct-Q4_K_M.gguf"),
        ;
        
        private final String org;
        private final String repo;
        private final String filename;
        
        TestModel(String org, String repo, String filename) {
            this.org = org;
            this.repo = repo;
            this.filename = filename;
        }
        
        public String getOrg() { return org; }
        public String getRepo() { return repo; }
        public String getFilename() { return filename; }
        
        public String getHuggingFaceUrl() {
            return String.format("https://huggingface.co/%s/%s/resolve/main/%s", 
                    org, repo, filename);
        }
        
        public String getCacheKey() {
            return String.format("%s_%s_%s", org, repo, filename.replace(".gguf", ""));
        }
    }
    
    public TestDataManager() {
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        this.cachePath = Paths.get(CACHE_DIR);
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
            System.out.println("Loading cached metadata for: " + model.name());
            GGUF gguf = loadFromCache(cachedFile);
            loadedMetadata.put(cacheKey, gguf);
            return gguf;
        }
        
        // Download from Hugging Face
        System.out.println("Downloading metadata for: " + model.name());
        System.out.println("URL: " + model.getHuggingFaceUrl());
        downloadAndCache(model, cachedFile);
        
        GGUF gguf = loadFromCache(cachedFile);
        loadedMetadata.put(cacheKey, gguf);
        return gguf;
    }
    
    /**
     * Downloads only the metadata portion of a GGUF file.
     * Uses HTTP range requests to download just the header and metadata.
     */
    private void downloadAndCache(TestModel model, Path targetFile) 
            throws IOException, InterruptedException {
        String url = model.getHuggingFaceUrl();
        
        // Try downloading progressively larger chunks until we can parse the metadata
        int[] sizesToTry = {2, 5, 10, 20}; // MB
        byte[] data = null;
        
        for (int sizeMB : sizesToTry) {
            int sizeBytes = sizeMB * 1024 * 1024;
            System.out.println("  Trying to download first " + sizeMB + "MB...");
            
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Range", "bytes=0-" + (sizeBytes - 1))
                    .GET()
                    .build();
            
            HttpResponse<byte[]> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofByteArray());
            
            if (response.statusCode() != 200 && response.statusCode() != 206) {
                throw new IOException("Failed to download: HTTP " + response.statusCode());
            }
            
            data = response.body();
            
            // Verify it's a valid GGUF file by checking magic number
            if (data.length < 4 || 
                data[0] != 'G' || data[1] != 'G' || data[2] != 'U' || data[3] != 'F') {
                throw new IOException("Downloaded file is not a valid GGUF file (wrong magic number)");
            }
            
            // Try to parse it to see if we have enough data
            try {
                Path tempFile = Files.createTempFile("gguf-test-", ".partial");
                Files.write(tempFile, data);
                GGUF gguf = GGUF.read(tempFile);
                // If we get here, we have enough data
                Files.deleteIfExists(tempFile);
                System.out.println("  Successfully parsed metadata with " + sizeMB + "MB");
                break;
            } catch (Exception e) {
                System.out.println("  Need more data: " + e.getMessage());
                if (sizeMB == sizesToTry[sizesToTry.length - 1]) {
                    throw new IOException("Failed to download enough metadata after trying " + sizeMB + "MB", e);
                }
            }
        }
        
        // Save to cache
        Files.createDirectories(targetFile.getParent());
        Files.write(targetFile, data, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        
        System.out.println("Downloaded " + data.length + " bytes for " + model.name());
    }
    
    /**
     * Loads GGUF metadata from a cached partial file.
     */
    private GGUF loadFromCache(Path cachedFile) throws IOException {
        try (ReadableByteChannel channel = Files.newByteChannel(cachedFile, StandardOpenOption.READ)) {
            return GGUF.read(channel);
        }
    }
    
    /**
     * Clears all cached metadata files.
     */
    public void clearCache() throws IOException {
        if (Files.exists(cachePath)) {
            Files.walk(cachePath)
                    .sorted((a, b) -> -a.compareTo(b)) // Reverse order to delete files before dirs
                    .forEach(path -> {
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
    
    /**
     * Returns the path to the cache directory.
     */
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
        String architecture = gguf.getValueOrDefault(String.class, "general.architecture", "unknown");
        
        return new TokenizerMetadata(
                modelType, modelName, architecture, tokens, scores, tokenTypes, 
                merges, bosTokenId, eosTokenId, padTokenId, unkTokenId
        );
    }
    
    /**
     * Holder for tokenizer metadata extracted from GGUF.
     */
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
            Integer unkTokenId
    ) {
        public int vocabularySize() {
            return tokens != null ? tokens.length : 0;
        }
        
        public boolean isBpe() {
            return "gpt2".equalsIgnoreCase(modelType) || "bpe".equalsIgnoreCase(modelType);
        }
        
        public boolean isSentencePiece() {
            return "llama".equalsIgnoreCase(modelType) || "sp".equalsIgnoreCase(modelType) ||
                   "sentencepiece".equalsIgnoreCase(modelType);
        }
    }
}
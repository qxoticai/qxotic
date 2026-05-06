package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.testkit.TestCachePaths;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Manages downloading and caching of GGUF model metadata from Hugging Face. */
public class TestDataManager {

    private static final Path CACHE_DIR = TestCachePaths.resolveUnderTestArtifacts("gguf-metadata");

    private static final int MAX_RANGE_BYTES =
            Integer.getInteger("toknroll.test.gguf.maxMetadataBytes", 1024 << 20);

    private final HttpClient httpClient;
    private final Path cachePath;
    private final Map<String, GGUF> loadedMetadata;

    public enum TestModel {
        QWEN3_0_6B("Qwen", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        QWEN2_5_0_5B("Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        GEMMA_3_4B_UNSLOTH("unsloth", "gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
        MISTRAL_3_3B_BARTOWSKI(
                "bartowski",
                "mistralai_Ministral-3-3B-Instruct-2512-GGUF",
                "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf");

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
            return "https://huggingface.co/" + org + "/" + repo + "/resolve/main/" + filename;
        }

        public String getCacheKey() {
            return String.format("%s_%s_%s", org, repo, filename.replace(".gguf", ""));
        }
    }

    public TestDataManager() {
        this(CACHE_DIR);
    }

    public TestDataManager(Path cachePath) {
        this.httpClient =
                HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        this.cachePath = cachePath;
        this.loadedMetadata = new ConcurrentHashMap<>();

        try {
            Files.createDirectories(cachePath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create cache directory: " + cachePath, e);
        }
    }

    public GGUF getOrDownloadMetadata(TestModel model) throws IOException, InterruptedException {
        return getOrDownloadMetadata(model.getCacheKey(), model.getHuggingFaceUrl());
    }

    public GGUF getOrDownloadMetadata(String cacheKey, String url)
            throws IOException, InterruptedException {
        String stableKey = stableUrlCacheKey(url);
        String effectiveKey = cacheKey + ":" + stableKey;
        GGUF cached = loadedMetadata.get(effectiveKey);
        if (cached != null) {
            return cached;
        }

        Path cachedFile = cachePath.resolve(stableKey + ".gguf.metadata");
        if (Files.exists(cachedFile)) {
            try {
                GGUF gguf = loadFromCache(cachedFile);
                loadedMetadata.put(effectiveKey, gguf);
                return gguf;
            } catch (IOException staleOrCorruptCache) {
                Files.deleteIfExists(cachedFile);
            }
        }

        downloadMetadata(url, cachedFile);
        GGUF gguf = loadFromCache(cachedFile);
        loadedMetadata.put(effectiveKey, gguf);
        return gguf;
    }

    public Path getCachePath() {
        return cachePath;
    }

    public static String cacheFileNameForUrl(String url) {
        return stableUrlCacheKey(url) + ".gguf.metadata";
    }

    private void downloadMetadata(String url, Path outputPath)
            throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(URI.create(url)).GET().build();

        HttpResponse<InputStream> response =
                httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());

        int status = response.statusCode();
        if (status < 200 || status >= 300) {
            throw new IOException(
                    "Failed to download metadata from "
                            + url
                            + " (status: "
                            + status
                            + ")");
        }

        Path tempFile = Files.createTempFile("gguf-test-", ".metadata.partial");
        try (InputStream body = new BufferedInputStream(response.body(), 1 << 16);
                OutputStream tap =
                        new BufferedOutputStream(Files.newOutputStream(tempFile), 1 << 16)) {

            ReadableByteChannel sourceChannel = Channels.newChannel(body);
            TeeChannel teeChannel = new TeeChannel(sourceChannel, tap, MAX_RANGE_BYTES);

            GGUF.read(teeChannel);
            tap.flush();

            try {
                Files.move(
                        tempFile,
                        outputPath,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveFailure) {
                Files.move(
                        tempFile,
                        outputPath,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(tempFile);
            if (e instanceof IOException ioe) {
                throw ioe;
            }
            throw new IOException(
                    "Failed to parse GGUF metadata from " + url, e);
        }
    }

    private static final class TeeChannel implements ReadableByteChannel {
        private final ReadableByteChannel source;
        private final OutputStream tap;
        private final long maxBytes;
        private long totalBytes;
        private volatile boolean open = true;

        TeeChannel(ReadableByteChannel source, OutputStream tap, long maxBytes) {
            this.source = source;
            this.tap = tap;
            this.maxBytes = maxBytes;
        }

        @Override
        public int read(ByteBuffer dst) throws IOException {
            int pos = dst.position();
            int n = source.read(dst);
            if (n > 0) {
                totalBytes += n;
                if (totalBytes > maxBytes) {
                    throw new IOException(
                            "GGUF metadata exceeds maximum size of " + maxBytes + " bytes");
                }
                byte[] copy = new byte[n];
                dst.position(pos);
                dst.get(copy);
                tap.write(copy);
            }
            return n;
        }

        @Override
        public boolean isOpen() {
            return open;
        }

        @Override
        public void close() throws IOException {
            open = false;
        }
    }

    private GGUF loadFromCache(Path cachedFile) throws IOException {
        try (java.nio.channels.FileChannel channel =
                java.nio.channels.FileChannel.open(cachedFile, StandardOpenOption.READ)) {
            return GGUF.read(channel);
        }
    }

    private static String stableUrlCacheKey(String url) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(url.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(hash.length * 2);
            for (byte b : hash) {
                sb.append(Character.forDigit((b >>> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    public static TokenizerMetadata extractTokenizerMetadata(GGUF gguf) {
        String modelType = gguf.getValueOrDefault(String.class, "tokenizer.ggml.model", "unknown");
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        float[] scores = gguf.getValueOrDefault(float[].class, "tokenizer.ggml.scores", null);
        String[] merges = gguf.getValueOrDefault(String[].class, "tokenizer.ggml.merges", null);
        int[] tokenTypesRaw =
                gguf.getValueOrDefault(int[].class, "tokenizer.ggml.token_type", null);
        int[] tokenTypes = new int[tokens.length];
        if (tokenTypesRaw != null) {
            for (int i = 0; i < Math.min(tokenTypesRaw.length, tokenTypes.length); i++) {
                tokenTypes[i] = tokenTypesRaw[i];
            }
        }

        Integer bos = gguf.getValueOrDefault(Integer.class, "tokenizer.ggml.bos_token_id", null);
        Integer eos = gguf.getValueOrDefault(Integer.class, "tokenizer.ggml.eos_token_id", null);
        Integer pad =
                gguf.getValueOrDefault(Integer.class, "tokenizer.ggml.padding_token_id", null);
        String modelName = gguf.getValueOrDefault(String.class, "general.name", "unknown");

        return new TokenizerMetadata(
                modelName,
                modelType,
                tokens.length,
                tokens,
                tokenTypes,
                scores,
                merges,
                bos,
                eos,
                pad);
    }

    public static final class TokenizerMetadata {
        private final String modelName;
        private final String modelType;
        private final int vocabularySize;
        private final String[] tokens;
        private final int[] tokenTypes;
        private final float[] scores;
        private final String[] merges;
        private final Integer bosTokenId;
        private final Integer eosTokenId;
        private final Integer padTokenId;

        public TokenizerMetadata(
                String modelName,
                String modelType,
                int vocabularySize,
                String[] tokens,
                int[] tokenTypes,
                float[] scores,
                String[] merges,
                Integer bosTokenId,
                Integer eosTokenId,
                Integer padTokenId) {
            this.modelName = modelName;
            this.modelType = modelType;
            this.vocabularySize = vocabularySize;
            this.tokens = tokens;
            this.tokenTypes = tokenTypes;
            this.scores = scores;
            this.merges = merges;
            this.bosTokenId = bosTokenId;
            this.eosTokenId = eosTokenId;
            this.padTokenId = padTokenId;
        }

        public String modelName() {
            return modelName;
        }

        public String modelType() {
            return modelType;
        }

        public int vocabularySize() {
            return vocabularySize;
        }

        public String[] tokens() {
            return tokens;
        }

        public int[] tokenTypes() {
            return tokenTypes;
        }

        public float[] scores() {
            return scores;
        }

        public String[] merges() {
            return merges;
        }

        public Integer bosTokenId() {
            return bosTokenId;
        }

        public Integer eosTokenId() {
            return eosTokenId;
        }

        public Integer padTokenId() {
            return padTokenId;
        }

        public boolean isBpe() {
            String type = modelType == null ? "" : modelType.toLowerCase();
            String name = modelName == null ? "" : modelName.toLowerCase();
            if (name.contains("gemma")) {
                return false;
            }
            return type.contains("bpe")
                    || type.contains("gpt")
                    || type.contains("qwen")
                    || type.contains("llama");
        }

        public boolean isSentencePiece() {
            String type = modelType == null ? "" : modelType.toLowerCase();
            String name = modelName == null ? "" : modelName.toLowerCase();
            return type.contains("spm")
                    || type.contains("sentencepiece")
                    || type.contains("gemma")
                    || name.contains("gemma");
        }
    }
}

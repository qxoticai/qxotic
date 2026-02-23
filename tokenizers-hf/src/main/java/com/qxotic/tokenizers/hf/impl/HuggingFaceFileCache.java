package com.qxotic.tokenizers.hf.impl;

import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

/**
 * Downloads and caches HuggingFace tokenizer files.
 *
 * <p>Caches files in {@code ~/.cache/qxotic-tokenizers/huggingface/} by default. Supports
 * environment variable overrides for cache directory and offline mode.
 */
public class HuggingFaceFileCache {

    private static final String DEFAULT_CACHE_DIR =
            System.getProperty("user.home") + "/.cache/qxotic-tokenizers/huggingface";
    private static final String CACHE_DIR_ENV = "QXOTIC_TOKENIZERS_HF_CACHE_DIR";
    private static final String OFFLINE_ENV = "QXOTIC_TOKENIZERS_HF_OFFLINE";
    private static final String HF_TOKEN_ENV = "QXOTIC_HF_TOKEN";

    private final HttpClient httpClient;
    private final Path cachePath;
    private final boolean offline;

    public HuggingFaceFileCache() {
        this.httpClient =
                HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        this.cachePath = resolveCachePath();
        this.offline = Boolean.parseBoolean(System.getenv(OFFLINE_ENV));
        ensureCacheDirectory();
    }

    /**
     * Gets or downloads a file from a HuggingFace repository.
     *
     * @param repoId the repository ID (e.g., "Qwen/Qwen3-0.6B")
     * @param revision the revision (e.g., "main")
     * @param filename the filename to download
     * @return path to the local file
     * @throws HuggingFaceTokenizerException if download fails or offline mode prevents download
     */
    public Path getOrDownload(String repoId, String revision, String filename) {
        Path localPath = getLocalPath(repoId, revision, filename);

        // Check if already cached
        if (Files.exists(localPath)) {
            return localPath;
        }

        if (offline) {
            throw new HuggingFaceTokenizerException(
                    "File not found in cache and offline mode is enabled: "
                            + filename
                            + "\n"
                            + "Expected at: "
                            + localPath
                            + "\n"
                            + "Set "
                            + OFFLINE_ENV
                            + "=false or download the file first.");
        }

        // Download from HuggingFace
        download(repoId, revision, filename, localPath);
        return localPath;
    }

    /** Downloads a file from HuggingFace Hub to the local cache. */
    private void download(String repoId, String revision, String filename, Path targetPath) {
        String url = buildUrl(repoId, revision, filename);

        try {
            // Ensure parent directory exists
            Files.createDirectories(targetPath.getParent());

            HttpRequest.Builder requestBuilder =
                    HttpRequest.newBuilder().uri(URI.create(url)).GET();

            // Add authorization header if token is set
            String token = System.getenv(HF_TOKEN_ENV);
            if (token != null && !token.isEmpty()) {
                requestBuilder.header("Authorization", "Bearer " + token);
            }

            HttpResponse<byte[]> response =
                    httpClient.send(
                            requestBuilder.build(), HttpResponse.BodyHandlers.ofByteArray());

            if (response.statusCode() == 401) {
                throw new HuggingFaceTokenizerException(
                        "Authentication required for repository: "
                                + repoId
                                + "\n"
                                + "Set "
                                + HF_TOKEN_ENV
                                + " environment variable with your HuggingFace token.");
            }

            if (response.statusCode() != 200) {
                throw new HuggingFaceTokenizerException(
                        "HTTP "
                                + response.statusCode()
                                + " error downloading "
                                + filename
                                + " from "
                                + repoId
                                + "\n"
                                + "URL: "
                                + url);
            }

            // Write atomically using temp file
            Path tempPath =
                    targetPath.getParent().resolve(filename + "." + UUID.randomUUID() + ".tmp");
            try {
                Files.write(tempPath, response.body());
                Files.move(
                        tempPath,
                        targetPath,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                Files.deleteIfExists(tempPath);
                throw e;
            }

        } catch (IOException e) {
            throw new HuggingFaceTokenizerException(
                    "Failed to download " + filename + " from " + repoId, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new HuggingFaceTokenizerException(
                    "Download interrupted for " + filename + " from " + repoId, e);
        }
    }

    /** Builds the HuggingFace Hub download URL. */
    private String buildUrl(String repoId, String revision, String filename) {
        return String.format("https://huggingface.co/%s/resolve/%s/%s", repoId, revision, filename);
    }

    /** Gets the local cache path for a file. */
    private Path getLocalPath(String repoId, String revision, String filename) {
        String safeRepoId = repoId.replace('/', '_');
        return cachePath.resolve(safeRepoId).resolve(revision).resolve(filename);
    }

    /** Returns the cache directory path. */
    private static Path resolveCachePath() {
        String cacheDir = System.getenv(CACHE_DIR_ENV);
        if (cacheDir != null && !cacheDir.isEmpty()) {
            return Paths.get(cacheDir);
        }
        return Paths.get(DEFAULT_CACHE_DIR);
    }

    /** Ensures the cache directory exists. */
    private void ensureCacheDirectory() {
        try {
            Files.createDirectories(cachePath);
        } catch (IOException e) {
            throw new HuggingFaceTokenizerException(
                    "Failed to create cache directory: " + cachePath, e);
        }
    }

    /** Returns the cache directory path. */
    public Path getCacheDirectory() {
        return cachePath;
    }
}

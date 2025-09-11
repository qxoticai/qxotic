package com.llm4j.tokenizers.impl;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;

public class FileCache {
    private static final String DEFAULT_CACHE_DIR = "data-gym-cache";

    /**
     * Reads file content from URL or filesystem with optional caching
     */
        public static byte[] readFileCached(String blobPath, String expectedHash) throws IOException, InterruptedException {
        String cacheDir = getCacheDir();
        if (cacheDir.isEmpty()) {
            return readFile(blobPath);
        }

        String cacheKey = sha1(blobPath);
        Path cachePath = Paths.get(cacheDir, cacheKey);

        // Try reading from cache first
        if (Files.exists(cachePath)) {
            byte[] data = Files.readAllBytes(cachePath);
            if (expectedHash == null || checkHash(data, expectedHash)) {
                return data;
            }

            // Invalid hash, remove cached file
            try {
                Files.delete(cachePath);
            } catch (IOException ignored) {
            }
        }

        // Download/read fresh content
        byte[] contents = readFile(blobPath);
        if (expectedHash != null && !checkHash(contents, expectedHash)) {
            throw new IOException(String.format(
                    "Hash mismatch for data from %s (expected %s). This may indicate corruption.",
                    blobPath, expectedHash));
        }

        // Cache the contents
        boolean userSpecifiedCache = isUserSpecifiedCache();
        try {
            Files.createDirectories(Paths.get(cacheDir));
            Path tmpPath = Paths.get(cacheDir,
                    cacheKey + "." + UUID.randomUUID() + ".tmp");
            Files.write(tmpPath, contents);
            Files.move(tmpPath, cachePath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            if (userSpecifiedCache) {
                throw e;
            }
            // Ignore cache write failures for default cache
        }

        return contents;
    }

    private static byte[] readFile(String blobPath) throws IOException, InterruptedException {
        if (!blobPath.startsWith("http://") && !blobPath.startsWith("https://")) {
            return Files.readAllBytes(Paths.get(blobPath));
        }

        // HTTP(S) download
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(blobPath))
                .build();

        HttpResponse<byte[]> response = client.send(request,
                HttpResponse.BodyHandlers.ofByteArray());

        if (response.statusCode() != 200) {
            throw new IOException("HTTP " + response.statusCode() +
                    " error downloading " + blobPath);
        }

        return response.body();
    }

    private static boolean checkHash(byte[] data, String expectedHash) {
        String actualHash = sha256(data);
        return actualHash.equals(expectedHash);
    }

    private static String getCacheDir() {
        String cacheDir = System.getenv("TIKTOKEN_CACHE_DIR");
        if (cacheDir != null) return cacheDir;

        cacheDir = System.getenv("DATA_GYM_CACHE_DIR");
        if (cacheDir != null) return cacheDir;

        return Paths.get(System.getProperty("java.io.tmpdir"),
                DEFAULT_CACHE_DIR).toString();
    }

    private static boolean isUserSpecifiedCache() {
        return System.getenv("TIKTOKEN_CACHE_DIR") != null ||
                System.getenv("DATA_GYM_CACHE_DIR") != null;
    }

    private static String sha1(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-1");
            byte[] hash = md.digest(input.getBytes());
            return bytesToHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-1 not available", e);
        }
    }

    private static String sha256(byte[] input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(input);
            return bytesToHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder hex = new StringBuilder();
        for (byte b : bytes) {
            hex.append(String.format("%02x", b));
        }
        return hex.toString();
    }
}

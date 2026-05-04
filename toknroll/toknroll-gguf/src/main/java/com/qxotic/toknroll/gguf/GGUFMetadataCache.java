package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

final class GGUFMetadataCache {
    private static final String ROOT_PROPERTY = "toknroll.cache.root";
    private static final String ROOT_ENV = "TOKNROLL_CACHE_ROOT";
    private static final String DEFAULT_CACHE_DIR = "cache";

    private static final String HF_TOKEN_PROPERTY = "toknroll.huggingface.token";
    private static final String HF_TOKEN_ENV = "HF_TOKEN";
    private static final String MODELSCOPE_TOKEN_PROPERTY = "toknroll.modelscope.token";
    private static final String MODELSCOPE_TOKEN_ENV = "MODELSCOPE_TOKEN";

    private static final int INITIAL_RANGE_BYTES = 1 << 20;
    private static final int MAX_RANGE_BYTES =
            Integer.getInteger("toknroll.gguf.maxMetadataBytes", 1024 << 20);

    private final Path cacheRoot;
    private volatile HttpClient httpClient;

    private GGUFMetadataCache(Path cacheRoot) {
        this.cacheRoot = cacheRoot;
    }

    static GGUFMetadataCache create() {
        return new GGUFMetadataCache(resolveCacheRoot());
    }

    static GGUFMetadataCache create(Path cacheRoot) {
        return new GGUFMetadataCache(cacheRoot.toAbsolutePath().normalize());
    }

    Path fetchHuggingFace(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        String resolvedRevision = normalizeHuggingFaceRevision(revision);
        String normalizedFile = normalizeFilePath(ggufPath);
        String url =
                "https://huggingface.co/"
                        + requireSegment(user, "user")
                        + "/"
                        + requireSegment(repository, "repository")
                        + "/resolve/"
                        + resolvedRevision
                        + "/"
                        + normalizedFile;
        Path base =
                cacheRoot
                        .resolve("gguf-metadata")
                        .resolve("huggingface")
                        .resolve(requireSegment(user, "user"))
                        .resolve(requireSegment(repository, "repository"))
                        .resolve(resolvedRevision);
        Path target = metadataFilePath(base, normalizedFile);
        String token = resolveToken(HF_TOKEN_PROPERTY, HF_TOKEN_ENV);
        return fetchPartialMetadata("huggingface", url, target, token, useCacheOnly, forceRefresh);
    }

    Path fetchModelScope(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        String resolvedRevision = normalizeModelScopeRevision(revision);
        String normalizedFile = normalizeFilePath(ggufPath);
        String url =
                "https://www.modelscope.cn/models/"
                        + requireSegment(user, "user")
                        + "/"
                        + requireSegment(repository, "repository")
                        + "/resolve/"
                        + resolvedRevision
                        + "/"
                        + normalizedFile;
        Path base =
                cacheRoot
                        .resolve("gguf-metadata")
                        .resolve("modelscope")
                        .resolve(requireSegment(user, "user"))
                        .resolve(requireSegment(repository, "repository"))
                        .resolve(resolvedRevision);
        Path target = metadataFilePath(base, normalizedFile);
        String token = resolveToken(MODELSCOPE_TOKEN_PROPERTY, MODELSCOPE_TOKEN_ENV);
        return fetchPartialMetadata("modelscope", url, target, token, useCacheOnly, forceRefresh);
    }

    private Path fetchPartialMetadata(
            String source,
            String url,
            Path target,
            String bearerToken,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        Path normalizedTarget = target.toAbsolutePath().normalize();
        if (!forceRefresh && Files.exists(normalizedTarget)) {
            return normalizedTarget;
        }
        if (useCacheOnly) {
            throw new IOException(
                    "[" + source + "] useCacheOnly=true and artifact not cached: " + url);
        }

        Files.createDirectories(normalizedTarget.getParent());
        IOException lastFailure = null;

        for (int range = INITIAL_RANGE_BYTES; range <= MAX_RANGE_BYTES; range <<= 1) {
            byte[] body;
            try {
                body = fetchRange(url, range, bearerToken, source);
            } catch (IOException e) {
                if (isHttp404(e)) {
                    throw e;
                }
                lastFailure = e;
                continue;
            }

            Path partial =
                    normalizedTarget.resolveSibling(
                            normalizedTarget.getFileName().toString() + ".partial");
            Files.write(partial, body);
            try {
                GGUF.read(partial);
                try {
                    Files.move(
                            partial,
                            normalizedTarget,
                            StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE);
                } catch (IOException atomicMoveFailure) {
                    Files.move(partial, normalizedTarget, StandardCopyOption.REPLACE_EXISTING);
                }
                return normalizedTarget;
            } catch (RuntimeException | IOException parseError) {
                lastFailure =
                        new IOException(
                                "["
                                        + source
                                        + "] Failed to parse GGUF metadata from partial download "
                                        + url
                                        + " (bytes="
                                        + body.length
                                        + ")",
                                parseError);
                Files.deleteIfExists(partial);
            }
        }

        if (lastFailure != null) {
            throw lastFailure;
        }
        throw new IOException("[" + source + "] Failed to fetch GGUF metadata: " + url);
    }

    private byte[] fetchRange(String url, int rangeBytes, String bearerToken, String source)
            throws IOException {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url)).GET();
        builder.header("Range", "bytes=0-" + (rangeBytes - 1));
        if (bearerToken != null && !bearerToken.isBlank()) {
            builder.header("Authorization", "Bearer " + bearerToken);
        }

        HttpResponse<byte[]> response;
        try {
            response =
                    getOrCreateHttpClient()
                            .send(builder.build(), HttpResponse.BodyHandlers.ofByteArray());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("[" + source + "] Interrupted while downloading " + url, e);
        }

        int status = response.statusCode();
        if ((status < 200 || status >= 300) && status != 206) {
            throw new IOException(
                    "[" + source + "] Failed to download " + url + " (HTTP " + status + ")");
        }
        return response.body();
    }

    private HttpClient getOrCreateHttpClient() {
        HttpClient client = httpClient;
        if (client == null) {
            synchronized (this) {
                client = httpClient;
                if (client == null) {
                    client =
                            HttpClient.newBuilder()
                                    .followRedirects(HttpClient.Redirect.NORMAL)
                                    .build();
                    httpClient = client;
                }
            }
        }
        return client;
    }

    private static boolean isHttp404(IOException e) {
        return e.getMessage() != null && e.getMessage().endsWith("(HTTP 404)");
    }

    private static String resolveToken(String propertyKey, String envKey) {
        String property = System.getProperty(propertyKey);
        if (property != null && !property.isBlank()) {
            return property;
        }
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            return env;
        }
        return null;
    }

    private static Path resolveCacheRoot() {
        String configured =
                firstNonBlank(System.getProperty(ROOT_PROPERTY), System.getenv(ROOT_ENV));
        if (configured != null) {
            return Path.of(configured).toAbsolutePath().normalize();
        }

        String home = System.getProperty("user.home", ".");
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isBlank()) {
                return Path.of(localAppData, "qxotic", "toknroll", DEFAULT_CACHE_DIR);
            }
            return Path.of(home, "AppData", "Local", "qxotic", "toknroll", DEFAULT_CACHE_DIR);
        }
        if (os.contains("mac")) {
            return Path.of(home, "Library", "Caches", "qxotic", "toknroll", DEFAULT_CACHE_DIR);
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        if (xdg != null && !xdg.isBlank()) {
            return Path.of(xdg, "qxotic", "toknroll", DEFAULT_CACHE_DIR);
        }
        return Path.of(home, ".cache", "qxotic", "toknroll", DEFAULT_CACHE_DIR);
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String normalizeHuggingFaceRevision(String revision) {
        if (revision == null) {
            return "main";
        }
        if (revision.isBlank()) {
            throw new IllegalArgumentException("revision must be null or non-blank");
        }
        return revision;
    }

    private static String normalizeModelScopeRevision(String revision) {
        if (revision == null) {
            return "master";
        }
        if (revision.isBlank()) {
            throw new IllegalArgumentException("revision must be null or non-blank");
        }
        return revision;
    }

    private static String requireSegment(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        if (value.contains("/") || value.contains("\\") || value.contains("..")) {
            throw new IllegalArgumentException(
                    name + " contains invalid path characters: " + value);
        }
        return value;
    }

    private static String normalizeFilePath(String file) {
        if (file == null || file.isBlank()) {
            throw new IllegalArgumentException("ggufPath must not be blank");
        }
        String normalized = file.replace('\\', '/');
        if (normalized.startsWith("/") || normalized.contains("..")) {
            throw new IllegalArgumentException("ggufPath contains invalid path segments: " + file);
        }
        if (!normalized.toLowerCase().endsWith(".gguf")) {
            throw new IllegalArgumentException("ggufPath must point to a .gguf file: " + file);
        }
        return normalized;
    }

    private static Path metadataFilePath(Path base, String ggufRelativePath) {
        Path out = base;
        for (String segment : (ggufRelativePath + ".metadata").split("/")) {
            if (segment.isEmpty() || ".".equals(segment) || "..".equals(segment)) {
                throw new IllegalArgumentException("Invalid ggufPath segment: " + ggufRelativePath);
            }
            out = out.resolve(segment);
        }
        return out;
    }
}

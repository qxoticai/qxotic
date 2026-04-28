package com.qxotic.toknroll.hf;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

final class RepositoryArtifactCache {
    private static final String ROOT_PROPERTY = "toknroll.cache.root";
    private static final String ROOT_ENV = "TOKNROLL_CACHE_ROOT";

    private static final String HF_TOKEN_PROPERTY = "toknroll.huggingface.token";
    private static final String HF_TOKEN_ENV = "HF_TOKEN";
    private static final String MODELSCOPE_TOKEN_PROPERTY = "toknroll.modelscope.token";
    private static final String MODELSCOPE_TOKEN_ENV = "MODELSCOPE_TOKEN";
    private static final String DEFAULT_CACHE_DIR = "cache";

    private final Path cacheRoot;
    private volatile HttpClient httpClient;

    private RepositoryArtifactCache(Path cacheRoot) {
        this.cacheRoot = cacheRoot;
    }

    static RepositoryArtifactCache create() {
        return new RepositoryArtifactCache(resolveCacheRoot());
    }

    static RepositoryArtifactCache create(Path cacheRoot) {
        return new RepositoryArtifactCache(cacheRoot.toAbsolutePath().normalize());
    }

    Path fetchUrl(
            String url,
            Map<String, List<String>> headers,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        Objects.requireNonNull(url, "url");
        String key = sha256(url);
        Path target =
                cacheRoot.resolve("repository-artifacts").resolve("url").resolve(key + ".bin");
        return fetchToPath("url", url, target, headers, useCacheOnly, forceRefresh);
    }

    /** Fetches and caches one file from a HuggingFace repository revision. */
    Path fetchHuggingFace(
            String user,
            String repository,
            String revision,
            String file,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        String resolvedRevision = normalizeRevision(revision);
        String url =
                "https://huggingface.co/"
                        + requireSegment(user, "user")
                        + "/"
                        + requireSegment(repository, "repository")
                        + "/resolve/"
                        + resolvedRevision
                        + "/"
                        + normalizeFilePath(file);

        Path target =
                cacheRoot
                        .resolve("repository-artifacts")
                        .resolve("huggingface")
                        .resolve(requireSegment(user, "user"))
                        .resolve(requireSegment(repository, "repository"))
                        .resolve(resolvedRevision);
        target = appendRelative(target, normalizeFilePath(file));

        return fetchToPath(
                "huggingface",
                url,
                target,
                authHeaders(resolveToken(HF_TOKEN_PROPERTY, HF_TOKEN_ENV)),
                useCacheOnly,
                forceRefresh);
    }

    /**
     * Fetches and caches one file from a ModelScope repository revision.
     *
     * <p>When {@code revision} is {@code null}, ModelScope defaults to {@code master}. Explicit
     * revision strings (including {@code main}) are not rewritten.
     */
    Path fetchModelScope(
            String user,
            String repository,
            String revision,
            String file,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        String resolvedRevision = normalizeModelScopeRevision(revision);
        String url =
                "https://www.modelscope.cn/models/"
                        + requireSegment(user, "user")
                        + "/"
                        + requireSegment(repository, "repository")
                        + "/resolve/"
                        + resolvedRevision
                        + "/"
                        + normalizeFilePath(file);

        Path target =
                cacheRoot
                        .resolve("repository-artifacts")
                        .resolve("modelscope")
                        .resolve(requireSegment(user, "user"))
                        .resolve(requireSegment(repository, "repository"))
                        .resolve(resolvedRevision);
        target = appendRelative(target, normalizeFilePath(file));

        return fetchToPath(
                "modelscope",
                url,
                target,
                authHeaders(resolveToken(MODELSCOPE_TOKEN_PROPERTY, MODELSCOPE_TOKEN_ENV)),
                useCacheOnly,
                forceRefresh);
    }

    private Path fetchToPath(
            String source,
            String url,
            Path target,
            Map<String, List<String>> headers,
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
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url)).GET();
        if (headers != null) {
            for (Map.Entry<String, List<String>> e : headers.entrySet()) {
                if (e.getValue() == null) {
                    continue;
                }
                for (String value : e.getValue()) {
                    if (value != null) {
                        builder.header(e.getKey(), value);
                    }
                }
            }
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

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException(
                    "["
                            + source
                            + "] Failed to download "
                            + url
                            + " (HTTP "
                            + response.statusCode()
                            + ")");
        }

        Path partial =
                normalizedTarget.resolveSibling(
                        normalizedTarget.getFileName().toString() + ".partial");
        Files.write(partial, response.body());
        Files.move(
                partial,
                normalizedTarget,
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE);
        return normalizedTarget;
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

    private static Map<String, List<String>> authHeaders(String token) {
        if (token == null || token.isBlank()) {
            return Map.of();
        }
        Map<String, List<String>> headers = new LinkedHashMap<>();
        headers.put("Authorization", List.of("Bearer " + token));
        return headers;
    }

    private static Path resolveCacheRoot() {
        String configured =
                firstNonBlank(System.getProperty(ROOT_PROPERTY), System.getenv(ROOT_ENV));
        if (configured != null) {
            return Path.of(configured).toAbsolutePath().normalize();
        }

        String home = System.getProperty("user.home", ".");
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
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

    private static String normalizeRevision(String revision) {
        if (revision == null) {
            return "main";
        }
        if (revision.isBlank()) {
            throw new IllegalArgumentException("revision must be null or non-blank");
        }
        return revision;
    }

    /**
     * Normalizes a ModelScope revision.
     *
     * <p>{@code null -> master}. Blank values are rejected.
     */
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
            throw new IllegalArgumentException("file must not be blank");
        }
        String normalized = file.replace('\\', '/');
        if (normalized.startsWith("/") || normalized.contains("..")) {
            throw new IllegalArgumentException("file contains invalid path segments: " + file);
        }
        return normalized;
    }

    private static Path appendRelative(Path root, String relativeFilePath) {
        Path out = root;
        for (String segment : relativeFilePath.split("/")) {
            if (segment.isEmpty() || ".".equals(segment) || "..".equals(segment)) {
                throw new IllegalArgumentException(
                        "Invalid file path segment: " + relativeFilePath);
            }
            out = out.resolve(segment);
        }
        return out;
    }

    private static String resolveToken(String propertyKey, String envKey) {
        return firstNonBlank(System.getProperty(propertyKey), System.getenv(envKey));
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(input.getBytes(UTF_8));
            StringBuilder sb = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                sb.append(Character.forDigit((b >>> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }
}

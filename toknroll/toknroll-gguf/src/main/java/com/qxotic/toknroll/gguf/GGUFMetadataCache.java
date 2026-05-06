package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;

final class GGUFMetadataCache {
    private static final String ROOT_PROPERTY = "toknroll.cache.root";
    private static final String ROOT_ENV = "TOKNROLL_CACHE_ROOT";
    private static final String DEFAULT_CACHE_DIR = "cache";

    private static final String HF_TOKEN_PROPERTY = "toknroll.huggingface.token";
    private static final String HF_TOKEN_ENV = "HF_TOKEN";
    private static final String MODELSCOPE_TOKEN_PROPERTY = "toknroll.modelscope.token";
    private static final String MODELSCOPE_TOKEN_ENV = "MODELSCOPE_TOKEN";

    private static final int MAX_RANGE_BYTES =
            Integer.getInteger("toknroll.gguf.maxMetadataBytes", 1024 << 20);

    private static final long CONNECT_TIMEOUT_SECONDS =
            Long.getLong("toknroll.gguf.connectTimeoutSeconds", 120);

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

    Path fetchPartialMetadata(
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

        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url)).GET();
        if (bearerToken != null && !bearerToken.isBlank()) {
            builder.header("Authorization", "Bearer " + bearerToken);
        }

        HttpResponse<InputStream> response;
        try {
            response =
                    getOrCreateHttpClient()
                            .send(builder.build(), HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("[" + source + "] Interrupted while downloading " + url, e);
        }

        Path partial =
                normalizedTarget.resolveSibling(
                        normalizedTarget.getFileName().toString() + ".partial");

        try (InputStream rawBody = response.body()) {
            int status = response.statusCode();
            if (status < 200 || status >= 300) {
                throw new IOException(
                        "[" + source + "] Failed to download " + url + " (HTTP " + status + ")");
            }

            InputStream body = new BufferedInputStream(rawBody, 1 << 16);
            OutputStream tap = new BufferedOutputStream(Files.newOutputStream(partial), 1 << 16);

            try {
                ReadableByteChannel sourceChannel = Channels.newChannel(body);
                TeeReadableByteChannel teeChannel =
                        new TeeReadableByteChannel(sourceChannel, tap, MAX_RANGE_BYTES);

                GGUF.read(teeChannel);
                tap.flush();
            } finally {
                tap.close();
            }

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
        } catch (IOException e) {
            Files.deleteIfExists(partial);
            throw e;
        } catch (RuntimeException e) {
            Files.deleteIfExists(partial);
            throw new IOException("[" + source + "] Failed to parse GGUF metadata from " + url, e);
        }
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
                                    .connectTimeout(Duration.ofSeconds(CONNECT_TIMEOUT_SECONDS))
                                    .build();
                    httpClient = client;
                }
            }
        }
        return client;
    }

    static final class TeeReadableByteChannel implements ReadableByteChannel {
        private final ReadableByteChannel source;
        private final OutputStream tap;
        private final long maxBytes;
        private long totalBytes;
        private volatile boolean open = true;

        TeeReadableByteChannel(ReadableByteChannel source, OutputStream tap, long maxBytes) {
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

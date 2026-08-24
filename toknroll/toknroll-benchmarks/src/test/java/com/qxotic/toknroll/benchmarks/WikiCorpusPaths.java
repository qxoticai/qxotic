package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.testkit.TestCachePaths;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipInputStream;

/**
 * Fixture locations for the enwik benchmark drivers. A corpus resolves from its
 * -Dtoknroll.enwikN.path override, the toknroll cache (see {@link TestCachePaths#cacheRoot()}), or
 * the legacy ~/.cache/qxotic/tokenizers/corpus location. Corpora are expensive (enwik9 is a 322 MB
 * zip), so nothing here downloads: a missing fixture fails loudly and names the explicit fetch
 * command, {@code make toknroll-fixtures} (see {@link FetchCorpus}). Driver outputs go to a
 * bench-output directory under the same root, never into the working tree.
 */
final class WikiCorpusPaths {

    /** Character range of the shared enwik9 slice fixture; the same constants the drivers use. */
    static final int SLICE_START_CHAR = 30_000_000;

    static final int SLICE_END_CHAR = 40_000_000;

    private static final List<String> NAMES = List.of("enwik8", "enwik9");
    private static final Map<String, String> URLS =
            Map.of(
                    "enwik8", "https://www.mattmahoney.net/dc/enwik8.zip",
                    "enwik9", "https://www.mattmahoney.net/dc/enwik9.zip");
    private static final Map<String, Long> EXPECTED_BYTES =
            Map.of("enwik8", 100_000_000L, "enwik9", 1_000_000_000L);

    private WikiCorpusPaths() {}

    static List<String> corpusNames() {
        return NAMES;
    }

    static long expectedBytes(String name) {
        Long bytes = EXPECTED_BYTES.get(name);
        if (bytes == null) {
            throw new IllegalArgumentException("Unsupported corpus: " + name);
        }
        return bytes;
    }

    static Path enwik8() throws IOException {
        return corpus("enwik8");
    }

    static Path enwik9() throws IOException {
        return corpus("enwik9");
    }

    static Path forCorpus(String corpus) throws IOException {
        if (!NAMES.contains(corpus)) {
            throw new IllegalArgumentException("Unsupported corpus: " + corpus);
        }
        return corpus(corpus);
    }

    /**
     * The 30M-40M character slice of enwik9, cached beside the corpus file. Derived from enwik9 on
     * first use, so there is no separate fixture to obtain; the derivation matches the in-memory
     * slicing the drivers do ({@link #SLICE_START_CHAR}..{@link #SLICE_END_CHAR}).
     */
    static Path slice30To40M() throws IOException {
        Path slice = enwik9().resolveSibling("slice_30_40m.txt");
        if (!Files.exists(slice)) {
            String text = Files.readString(enwik9());
            Files.writeString(slice, text.substring(SLICE_START_CHAR, SLICE_END_CHAR));
        }
        return slice;
    }

    /** A file under the bench-output directory (override: -Dtoknroll.bench.output.dir=...). */
    static Path benchOutput(String fileName) throws IOException {
        String configured = System.getProperty("toknroll.bench.output.dir");
        Path dir =
                configured == null || configured.isBlank()
                        ? TestCachePaths.cacheRoot().resolve("bench-output")
                        : Path.of(configured);
        Files.createDirectories(dir);
        return dir.resolve(fileName);
    }

    /** Where {@link FetchCorpus} puts a corpus. */
    static Path downloadTarget(String name) {
        return TestCachePaths.corpusDir().resolve(name);
    }

    /** The resolved corpus file, or null when none of the locations has it. */
    private static Path find(String name) {
        String configured = System.getProperty("toknroll." + name + ".path");
        if (configured != null && !configured.isBlank()) {
            Path configuredPath = Path.of(configured);
            if (Files.exists(configuredPath)) {
                return configuredPath;
            }
            throw new IllegalStateException(
                    "-Dtoknroll."
                            + name
                            + ".path points at a file that does not exist: "
                            + configuredPath);
        }
        Path cached = downloadTarget(name);
        if (Files.exists(cached)) {
            return cached;
        }
        Path legacy = legacy(name);
        return legacy.equals(cached) || !Files.exists(legacy) ? null : legacy;
    }

    private static Path legacy(String name) {
        return Path.of(System.getProperty("user.home"))
                .resolve(".cache/qxotic/tokenizers/corpus")
                .resolve(name);
    }

    private static Path corpus(String name) throws IOException {
        Path found = find(name);
        if (found != null) {
            return found;
        }
        throw new IOException(
                "Could not locate "
                        + name
                        + ". Fetch it with `make toknroll-fixtures` (or set -Dtoknroll."
                        + name
                        + ".path=/path/to/"
                        + name
                        + "). Checked: "
                        + downloadTarget(name)
                        + ", "
                        + legacy(name));
    }

    /**
     * Download the corpus zip, stream out its single entry, verify the size, rename into place. A
     * failed attempt deletes its partial file. Only {@link FetchCorpus} calls this: fixtures are
     * expensive, so fetching is an explicit command, never a side effect of a benchmark run.
     */
    static void download(String name, Path target) throws IOException, InterruptedException {
        String url = URLS.get(name);
        if (url == null) {
            throw new IllegalArgumentException("Unsupported corpus: " + name);
        }
        long expected = expectedBytes(name);
        Files.createDirectories(target.getParent());
        Path part = target.resolveSibling(name + ".part");
        System.out.printf("Downloading %s from %s ...%n", name, url);
        try {
            HttpResponse<InputStream> response =
                    HttpClient.newHttpClient()
                            .send(
                                    HttpRequest.newBuilder(URI.create(url)).build(),
                                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                throw new IOException(url + " answered HTTP " + response.statusCode());
            }
            long written;
            try (InputStream body = response.body();
                    ZipInputStream zip = new ZipInputStream(body)) {
                if (zip.getNextEntry() == null) {
                    throw new IOException(url + " is an empty zip");
                }
                written = Files.copy(zip, part, StandardCopyOption.REPLACE_EXISTING);
            }
            if (written != expected) {
                throw new IOException(name + " must be " + expected + " bytes, got " + written);
            }
            Files.move(part, target, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(part);
        }
    }
}

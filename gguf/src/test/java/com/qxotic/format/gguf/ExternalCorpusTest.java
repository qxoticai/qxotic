package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * External GGUF corpus validation against pinned Hugging Face artifacts.
 *
 * <p>These tests are opt-in and are skipped unless environment variable {@code GGUF_EXTERNAL_TESTS}
 * is set to {@code true}.
 *
 * <p>Run explicitly with:
 *
 * <pre>{@code
 * GGUF_EXTERNAL_TESTS=true mvnd -pl gguf -am test -Dtest=ExternalCorpusTest
 * }</pre>
 *
 * <p>The suite requires external model files from the network (Unsloth Hugging Face repositories)
 * and caches metadata-only GGUF snapshots locally.
 *
 * <p>Cache directory defaults to {@code ~/.cache/qxotic/gguf-metadata}. Override with {@code
 * GGUF_FIXTURES_DIR=/path/to/cache}.
 */
@Tag("external")
public class ExternalCorpusTest {

    private static final String ENABLE_EXTERNAL_TESTS_ENV = "GGUF_EXTERNAL_TESTS";
    private static final String FIXTURES_DIR_ENV = "GGUF_FIXTURES_DIR";

    private static final class CorpusEntry {
        private final String repo;
        private final String revision;
        private final String filename;
        private final String architecture;

        private CorpusEntry(String repo, String revision, String filename, String architecture) {
            this.repo = repo;
            this.revision = revision;
            this.filename = filename;
            this.architecture = architecture;
        }

        public String repo() {
            return this.repo;
        }

        public String revision() {
            return this.revision;
        }

        public String filename() {
            return this.filename;
        }

        public String architecture() {
            return this.architecture;
        }

        @Override
        public String toString() {
            return this.repo + "/" + this.filename + "@" + this.revision;
        }
    }

    static Stream<CorpusEntry> corpus() {
        return Stream.of(
                new CorpusEntry(
                        "unsloth/gpt-oss-20b-GGUF",
                        "d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4",
                        "gpt-oss-20b-Q8_0.gguf",
                        "gpt-oss"),
                new CorpusEntry(
                        "unsloth/gemma-4-26B-A4B-it-GGUF",
                        "8bacec5c8e829a25502cdfe3c3f5b6aabee3218c",
                        "gemma-4-26B-A4B-it-Q8_0.gguf",
                        "gemma4"),
                new CorpusEntry(
                        "unsloth/Qwen3.6-35B-A3B-GGUF",
                        "9280dd353ab587157920d5bd391ada414d84e552",
                        "Qwen3.6-35B-A3B-Q8_0.gguf",
                        "qwen35moe"));
    }

    @ParameterizedTest(name = "external corpus invariants: {0}")
    @MethodSource("corpus")
    public void testExternalCorpusInvariants(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        GGUF gguf = loadOrCreateCachedMetadata(entry);
        validateCorpusInvariants(gguf, entry.architecture());
    }

    @ParameterizedTest(name = "external corpus round-trip: {0}")
    @MethodSource("corpus")
    public void testMetadataRoundTrip(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        GGUF original = loadOrCreateCachedMetadata(entry);
        validateCorpusInvariants(original, entry.architecture());

        GGUF rebuilt = Builder.newBuilder(original).build(false);
        Path tempFile = Files.createTempFile("gguf-roundtrip-", ".gguf");
        Files.deleteIfExists(tempFile);
        try {
            GGUF.write(rebuilt, tempFile);
            GGUF reloaded = GGUF.read(tempFile);

            validateCorpusInvariants(reloaded, entry.architecture());
            assertEquivalentMetadataAndTensors(original, reloaded);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    private static void assumeExternalTestsEnabled() {
        Assumptions.assumeTrue(
                "true".equalsIgnoreCase(System.getenv(ENABLE_EXTERNAL_TESTS_ENV)),
                () -> "Set " + ENABLE_EXTERNAL_TESTS_ENV + "=true to run external corpus tests");
    }

    private static GGUF loadOrCreateCachedMetadata(CorpusEntry entry) throws IOException {
        Path cachedMetadataPath = cachedMetadataPath(entry);
        if (Files.exists(cachedMetadataPath)) {
            return GGUF.read(cachedMetadataPath);
        }

        Files.createDirectories(cachedMetadataPath.getParent());
        GGUF fetched = fetchFromHuggingFace(entry);
        GGUF metadataOnly = Builder.newBuilder(fetched).build(false);

        Path tmp =
                cachedMetadataPath
                        .getParent()
                        .resolve("gguf-meta-" + System.nanoTime() + ".tmp.gguf");
        try {
            GGUF.write(metadataOnly, tmp);
            Files.move(tmp, cachedMetadataPath, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(tmp);
        }
        return GGUF.read(cachedMetadataPath);
    }

    private static Path cachedMetadataPath(CorpusEntry entry) {
        String env = System.getenv(FIXTURES_DIR_ENV);
        Path baseDir;
        if (env != null && !env.isBlank()) {
            baseDir = Path.of(env);
        } else {
            baseDir = Path.of(System.getProperty("user.home"), ".cache", "qxotic", "gguf-metadata");
        }

        String[] repoParts = entry.repo().split("/");
        String filename = entry.filename().replace(".gguf", ".metadata.gguf");
        return baseDir.resolve(Path.of(repoParts[0], repoParts[1], entry.revision(), filename));
    }

    private static GGUF fetchFromHuggingFace(CorpusEntry entry) throws IOException {
        String url =
                String.format(
                        "https://huggingface.co/%s/resolve/%s/%s",
                        entry.repo(), entry.revision(), entry.filename());
        try (ReadableByteChannel channel =
                Channels.newChannel(new BufferedInputStream(new URL(url).openStream()))) {
            return GGUF.read(channel);
        }
    }

    private static void validateCorpusInvariants(GGUF gguf, String expectedArchitecture) {
        assertNotNull(gguf);
        assertFalse(gguf.getMetadataKeys().isEmpty(), "metadata should not be empty");
        assertFalse(gguf.getTensors().isEmpty(), "tensor list should not be empty");

        String architecture = gguf.getString("general.architecture");
        assertEquals(expectedArchitecture, architecture);

        int alignment = gguf.getAlignment();
        assertTrue(alignment > 0, "alignment must be positive");
        assertEquals(1, Integer.bitCount(alignment), "alignment should be power-of-two");

        Set<String> tensorNames = new HashSet<>();
        long previousOffset = -1;
        long previousEnd = -1;
        boolean hasQ80Tensor = false;

        for (TensorEntry tensor : gguf.getTensors()) {
            assertNotNull(tensor);
            assertNotNull(tensor.name());
            assertFalse(tensor.name().isEmpty(), "tensor name should not be empty");
            assertTrue(
                    tensorNames.add(tensor.name()),
                    "tensor names must be unique: " + tensor.name());

            long[] shape = tensor.shape();
            assertTrue(shape.length > 0, "tensor rank should be positive: " + tensor.name());
            for (long dim : shape) {
                assertTrue(dim > 0, "tensor dimensions must be positive");
            }

            long offset = tensor.offset();
            long byteSize = tensor.byteSize();
            assertTrue(offset >= 0, "offset should be non-negative");
            assertTrue(byteSize > 0, "tensor byte size should be positive");
            assertEquals(0L, offset % alignment, "tensor offset should satisfy file alignment");
            assertTrue(offset >= previousOffset, "tensor offsets should be non-decreasing");
            assertTrue(offset >= previousEnd, "tensor data should not overlap");

            long totalElements = tensor.totalNumberOfElements();
            assertEquals(totalElements, tensor.ggmlType().elementsForByteSize(byteSize));

            previousOffset = offset;
            previousEnd = Math.addExact(offset, byteSize);

            if (tensor.ggmlType() == GGMLType.Q8_0) {
                hasQ80Tensor = true;
            }
        }

        assertTrue(hasQ80Tensor, "expected at least one Q8_0 tensor in corpus entry");
    }

    private static void assertEquivalentMetadataAndTensors(GGUF expected, GGUF actual) {
        assertEquals(expected.getVersion(), actual.getVersion());
        assertEquals(expected.getAlignment(), actual.getAlignment());
        assertEquals(expected.getMetadataKeys(), actual.getMetadataKeys());

        for (String key : expected.getMetadataKeys()) {
            assertEquals(
                    expected.getType(key),
                    actual.getType(key),
                    "metadata type mismatch for key " + key);
            assertEquals(
                    expected.getComponentType(key),
                    actual.getComponentType(key),
                    "metadata component type mismatch for key " + key);
            assertMetadataValueEquals(
                    expected.getValue(Object.class, key),
                    actual.getValue(Object.class, key),
                    "metadata value mismatch for key " + key);
        }

        Collection<TensorEntry> expectedTensors = expected.getTensors();
        Collection<TensorEntry> actualTensors = actual.getTensors();
        assertEquals(expectedTensors.size(), actualTensors.size());

        Map<String, TensorEntry> expectedByName =
                expectedTensors.stream()
                        .collect(Collectors.toMap(TensorEntry::name, Function.identity()));
        Map<String, TensorEntry> actualByName =
                actualTensors.stream()
                        .collect(Collectors.toMap(TensorEntry::name, Function.identity()));

        assertEquals(expectedByName.keySet(), actualByName.keySet());

        for (String tensorName : expectedByName.keySet()) {
            TensorEntry e = expectedByName.get(tensorName);
            TensorEntry a = actualByName.get(tensorName);
            assertArrayEquals(e.shape(), a.shape(), "shape mismatch for tensor " + tensorName);
            assertEquals(e.ggmlType(), a.ggmlType(), "type mismatch for tensor " + tensorName);
            assertEquals(e.offset(), a.offset(), "offset mismatch for tensor " + tensorName);
            assertEquals(e.byteSize(), a.byteSize(), "byte size mismatch for tensor " + tensorName);
        }
    }

    private static void assertMetadataValueEquals(Object expected, Object actual, String message) {
        if (expected == null || actual == null) {
            assertEquals(expected, actual, message);
            return;
        }

        Class<?> expectedClass = expected.getClass();
        Class<?> actualClass = actual.getClass();
        assertEquals(expectedClass, actualClass, message + " (class mismatch)");

        if (!expectedClass.isArray()) {
            assertEquals(expected, actual, message);
            return;
        }

        if (expected instanceof byte[] && actual instanceof byte[]) {
            byte[] a = (byte[]) expected;
            byte[] b = (byte[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof short[] && actual instanceof short[]) {
            short[] a = (short[]) expected;
            short[] b = (short[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof int[] && actual instanceof int[]) {
            int[] a = (int[]) expected;
            int[] b = (int[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof long[] && actual instanceof long[]) {
            long[] a = (long[]) expected;
            long[] b = (long[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof float[] && actual instanceof float[]) {
            float[] a = (float[]) expected;
            float[] b = (float[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof double[] && actual instanceof double[]) {
            double[] a = (double[]) expected;
            double[] b = (double[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof boolean[] && actual instanceof boolean[]) {
            boolean[] a = (boolean[]) expected;
            boolean[] b = (boolean[]) actual;
            assertArrayEquals(a, b, message);
            return;
        }
        if (expected instanceof Object[] && actual instanceof Object[]) {
            Object[] a = (Object[]) expected;
            Object[] b = (Object[]) actual;
            assertTrue(Arrays.deepEquals(a, b), message);
            return;
        }

        assertEquals(expected, actual, message);
    }
}

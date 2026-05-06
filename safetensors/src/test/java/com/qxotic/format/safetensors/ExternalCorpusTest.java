package com.qxotic.format.safetensors;

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
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * External Safetensors corpus validation against pinned Hugging Face artifacts.
 *
 * <p>These tests are opt-in and are skipped unless environment variable {@code
 * SAFETENSORS_EXTERNAL_TESTS} is set to {@code true}.
 *
 * <p>Run explicitly with:
 *
 * <pre>{@code
 * SAFETENSORS_EXTERNAL_TESTS=true mvn -pl safetensors -am test -Dtest=ExternalCorpusTest
 * }</pre>
 *
 * <p>Cache directory defaults to {@code ~/.cache/qxotic/safetensors-metadata}. Override with {@code
 * SAFETENSORS_FIXTURES_DIR=/path/to/cache}.
 */
@Tag("external")
public class ExternalCorpusTest {

    private static final String ENABLE_EXTERNAL_TESTS_ENV = "SAFETENSORS_EXTERNAL_TESTS";
    private static final String FIXTURES_DIR_ENV = "SAFETENSORS_FIXTURES_DIR";

    private static final class CorpusEntry {
        private final String repo;
        private final String revision;
        private final String filename;
        private final String expectedTensor;
        private final int minTensorCount;

        private CorpusEntry(
                String repo,
                String revision,
                String filename,
                String expectedTensor,
                int minTensorCount) {
            this.repo = repo;
            this.revision = revision;
            this.filename = filename;
            this.expectedTensor = expectedTensor;
            this.minTensorCount = minTensorCount;
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

        public String expectedTensor() {
            return this.expectedTensor;
        }

        public int minTensorCount() {
            return this.minTensorCount;
        }

        @Override
        public String toString() {
            return this.repo + "/" + this.filename + "@" + this.revision;
        }
    }

    static Stream<CorpusEntry> corpus() {
        return Stream.of(
                new CorpusEntry(
                        "google/gemma-4-26B-A4B-it",
                        "7d4c97e54145f8ffd1a4dd1b4986a5015a517842",
                        "model-00001-of-00002.safetensors",
                        "model.language_model.embed_tokens.weight",
                        900),
                new CorpusEntry(
                        "google/gemma-4-E2B-it",
                        "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
                        "model.safetensors",
                        "model.language_model.embed_tokens.weight",
                        1800),
                new CorpusEntry(
                        "Qwen/Qwen3.6-35B-A3B",
                        "7da1103448ba36029c34ce1a9a741dfe93ee0c50",
                        "model-00001-of-00026.safetensors",
                        "model.language_model.embed_tokens.weight",
                        20),
                new CorpusEntry(
                        "HuggingFaceTB/SmolLM3-3B",
                        "a07cc9a04f16550a088caea529712d1d335b0ac1",
                        "model-00001-of-00002.safetensors",
                        "model.embed_tokens.weight",
                        200));
    }

    @ParameterizedTest(name = "external corpus invariants: {0}")
    @MethodSource("corpus")
    public void testExternalCorpusInvariants(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        Safetensors st = loadOrCreateCachedMetadata(entry);
        validateCorpusInvariants(st, entry);
    }

    @ParameterizedTest(name = "external corpus round-trip: {0}")
    @MethodSource("corpus")
    public void testMetadataRoundTrip(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        Safetensors original = loadOrCreateCachedMetadata(entry);
        validateCorpusInvariants(original, entry);

        Safetensors rebuilt = Builder.newBuilder(original).build(false);
        Path tempFile = Files.createTempFile("safetensors-roundtrip-", ".safetensors");
        Files.deleteIfExists(tempFile);
        try {
            Safetensors.write(rebuilt, tempFile);
            Safetensors reloaded = Safetensors.read(tempFile);

            validateCorpusInvariants(reloaded, entry);
            SafetensorsTest.assertEqualsSafetensors(original, reloaded, true);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    private static void assumeExternalTestsEnabled() {
        Assumptions.assumeTrue(
                "true".equalsIgnoreCase(System.getenv(ENABLE_EXTERNAL_TESTS_ENV)),
                () -> "Set " + ENABLE_EXTERNAL_TESTS_ENV + "=true to run external corpus tests");
    }

    private static Safetensors loadOrCreateCachedMetadata(CorpusEntry entry) throws IOException {
        Path cachedMetadataPath = cachedMetadataPath(entry);
        if (Files.exists(cachedMetadataPath)) {
            return Safetensors.read(cachedMetadataPath);
        }

        Files.createDirectories(cachedMetadataPath.getParent());
        Safetensors fetched = fetchFromHuggingFace(entry);
        Safetensors metadataOnly = Builder.newBuilder(fetched).build(false);

        Path tmp =
                cachedMetadataPath
                        .getParent()
                        .resolve("safetensors-meta-" + System.nanoTime() + ".tmp.safetensors");
        try {
            Safetensors.write(metadataOnly, tmp);
            Files.move(tmp, cachedMetadataPath, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(tmp);
        }
        return Safetensors.read(cachedMetadataPath);
    }

    private static Path cachedMetadataPath(CorpusEntry entry) {
        String env = System.getenv(FIXTURES_DIR_ENV);
        Path baseDir;
        if (env != null && !env.isBlank()) {
            baseDir = Path.of(env);
        } else {
            baseDir =
                    Path.of(
                            System.getProperty("user.home"),
                            ".cache",
                            "qxotic",
                            "safetensors-metadata");
        }

        String[] repoParts = entry.repo().split("/");
        String filename = entry.filename().replace(".safetensors", ".metadata.safetensors");
        return baseDir.resolve(
                Path.of(repoParts[0], repoParts[repoParts.length - 1], entry.revision(), filename));
    }

    private static Safetensors fetchFromHuggingFace(CorpusEntry entry) throws IOException {
        String url =
                String.format(
                        "https://huggingface.co/%s/resolve/%s/%s",
                        entry.repo(), entry.revision(), entry.filename());
        try (ReadableByteChannel channel =
                Channels.newChannel(new BufferedInputStream(new URL(url).openStream()))) {
            return Safetensors.read(channel);
        }
    }

    private static void validateCorpusInvariants(Safetensors st, CorpusEntry entry) {
        assertNotNull(st);
        assertTrue(st.getAlignment() > 0, "alignment must be positive");
        assertEquals(1, Integer.bitCount(st.getAlignment()), "alignment should be power-of-two");
        assertFalse(st.getTensors().isEmpty(), "tensor list should not be empty");
        assertTrue(
                st.getTensors().size() >= entry.minTensorCount(),
                "unexpectedly small tensor count");

        String format = st.getMetadata().get("format");
        assertEquals("pt", format, "expected metadata format=pt");

        TensorEntry expected = st.getTensor(entry.expectedTensor());
        assertNotNull(expected, "expected tensor not found: " + entry.expectedTensor());

        Set<String> tensorNames = new HashSet<>();
        long previousOffset = -1;
        long previousEnd = -1;

        for (TensorEntry tensor : st.getTensors()) {
            assertNotNull(tensor);
            assertNotNull(tensor.name());
            assertFalse(tensor.name().isEmpty(), "tensor name should not be empty");
            assertTrue(
                    tensorNames.add(tensor.name()),
                    "tensor names must be unique: " + tensor.name());

            assertNotNull(tensor.dtype(), "dtype must be present");
            for (long dim : tensor.shape()) {
                assertTrue(dim >= 0, "tensor dimensions must be non-negative");
            }

            long offset = tensor.byteOffset();
            long byteSize = tensor.byteSize();
            assertTrue(offset >= 0, "offset should be non-negative");
            assertTrue(byteSize >= 0, "tensor byte size should be non-negative");
            assertTrue(offset >= previousOffset, "tensor offsets should be non-decreasing");
            assertTrue(offset >= previousEnd, "tensor data should not overlap");

            previousOffset = offset;
            previousEnd = Math.addExact(offset, byteSize);
        }
    }
}

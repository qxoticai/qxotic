package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * External JSON corpus validation against pinned upstream test suites.
 *
 * <p>These tests are opt-in and are skipped unless environment variable {@code JSON_EXTERNAL_TESTS}
 * is set to {@code true}.
 *
 * <p>Run explicitly with:
 *
 * <pre>{@code
 * JSON_EXTERNAL_TESTS=true mvn -pl json -am test -Dtest=ExternalCorpusTest
 * }</pre>
 *
 * <p>Cache directory defaults to {@code ~/.cache/qxotic/json-corpora}. Override with {@code
 * JSON_FIXTURES_DIR=/path/to/cache}.
 */
@Tag("external")
class ExternalCorpusTest {

    private static final String ENABLE_EXTERNAL_TESTS_ENV = "JSON_EXTERNAL_TESTS";
    private static final String FIXTURES_DIR_ENV = "JSON_FIXTURES_DIR";

    private static final class CorpusEntry {
        private final String repo;
        private final String revision;

        private CorpusEntry(String repo, String revision) {
            this.repo = repo;
            this.revision = revision;
        }

        public String repo() {
            return this.repo;
        }

        public String revision() {
            return this.revision;
        }

        @Override
        public String toString() {
            return this.repo + "@" + this.revision;
        }
    }

    static Stream<CorpusEntry> corpus() {
        return Stream.of(
                new CorpusEntry("nst/JSONTestSuite", "1ef36fa01286573e846ac449e8683f8833c5b26a"));
    }

    @ParameterizedTest(name = "external corpus compliance: {0}")
    @MethodSource("corpus")
    void testExternalCorpusCompliance(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        Path suiteRoot = loadOrCreateCachedCorpus(entry);
        Path parsingDir = suiteRoot.resolve("test_parsing");

        Assumptions.assumeTrue(
                Files.isDirectory(parsingDir), () -> "test_parsing not found at: " + suiteRoot);

        CorpusRunSummary summary = runJsonTestSuiteParsingCorpus(parsingDir);
        System.out.println(summary.toSummaryString());

        assertEquals(0, summary.yFailed, () -> "Some y_ files were rejected:\n" + summary.failures);
        assertFalse(summary.nFailed > 0, () -> "Some n_ files were accepted:\n" + summary.failures);
    }

    @ParameterizedTest(name = "external corpus round-trip: {0}")
    @MethodSource("corpus")
    void testAcceptedRoundTrip(CorpusEntry entry) throws IOException {
        assumeExternalTestsEnabled();

        Path suiteRoot = loadOrCreateCachedCorpus(entry);
        Path parsingDir = suiteRoot.resolve("test_parsing");

        List<Path> acceptedFiles = jsonFilesByPrefix(parsingDir, "y_");
        Assumptions.assumeFalse(
                acceptedFiles.isEmpty(), () -> "No y_ files found in: " + parsingDir);

        StringBuilder failures = new StringBuilder();
        int failed = 0;

        for (Path file : acceptedFiles) {
            String name = file.getFileName().toString();
            byte[] bytes = Files.readAllBytes(file);

            String input;
            try {
                input = decodeStrictUtf8(bytes);
            } catch (CharacterCodingException e) {
                failed++;
                failures.append("Y round-trip failed (UTF-8 decode): ").append(name).append('\n');
                continue;
            }

            try {
                Object parsed = Json.parse(input);
                String reserialized = Json.stringify(parsed);
                Json.parse(reserialized);
            } catch (RuntimeException ex) {
                failed++;
                failures.append("Y round-trip failed: ")
                        .append(name)
                        .append(" -> ")
                        .append(ex.getClass().getSimpleName())
                        .append(": ")
                        .append(ex.getMessage())
                        .append('\n');
            }
        }

        assertEquals(0, failed, () -> "Some y_ files failed round-trip:\n" + failures);
    }

    private static void assumeExternalTestsEnabled() {
        Assumptions.assumeTrue(
                "true".equalsIgnoreCase(System.getenv(ENABLE_EXTERNAL_TESTS_ENV)),
                () -> "Set " + ENABLE_EXTERNAL_TESTS_ENV + "=true to run external corpus tests");
    }

    private static Path loadOrCreateCachedCorpus(CorpusEntry entry) throws IOException {
        Path suiteRoot = cachedSuiteRoot(entry);
        Path parsingDir = suiteRoot.resolve("test_parsing");
        if (Files.isDirectory(parsingDir)) {
            return suiteRoot;
        }

        Files.createDirectories(suiteRoot.getParent());

        Path stagingDir = suiteRoot.getParent().resolve("json-suite-stage-" + System.nanoTime());
        Path zipPath = suiteRoot.getParent().resolve("json-suite-" + System.nanoTime() + ".zip");
        Files.createDirectories(stagingDir);

        try {
            String url =
                    String.format(
                            Locale.ROOT,
                            "https://codeload.github.com/%s/zip/%s",
                            entry.repo(),
                            entry.revision());
            try (InputStream in = new BufferedInputStream(new java.net.URL(url).openStream())) {
                Files.copy(in, zipPath, StandardCopyOption.REPLACE_EXISTING);
            }

            unzip(zipPath, stagingDir);

            Path extractedRoot = singleExtractedDirectory(stagingDir);
            Path tmpTarget = suiteRoot.getParent().resolve("json-suite-tmp-" + System.nanoTime());
            Files.move(extractedRoot, tmpTarget, StandardCopyOption.REPLACE_EXISTING);
            Files.move(tmpTarget, suiteRoot, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(zipPath);
            deleteRecursivelyIfExists(stagingDir);
        }

        return suiteRoot;
    }

    private static Path cachedSuiteRoot(CorpusEntry entry) {
        String env = System.getenv(FIXTURES_DIR_ENV);
        Path baseDir;
        if (env != null && !env.isBlank()) {
            baseDir = Path.of(env);
        } else {
            baseDir = Path.of(System.getProperty("user.home"), ".cache", "qxotic", "json-corpora");
        }

        String[] parts = entry.repo().split("/");
        return baseDir.resolve(Path.of(parts[0], parts[1], entry.revision()));
    }

    private static void unzip(Path zipPath, Path destinationRoot) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipPath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.isDirectory()) {
                    continue;
                }

                Path destinationPath = destinationRoot.resolve(entry.getName()).normalize();
                if (!destinationPath.startsWith(destinationRoot)) {
                    throw new IOException("Zip entry escapes destination: " + entry.getName());
                }

                Files.createDirectories(destinationPath.getParent());
                Files.copy(zis, destinationPath, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private static Path singleExtractedDirectory(Path stagingDir) throws IOException {
        List<Path> roots;
        try (Stream<Path> stream = Files.list(stagingDir)) {
            roots = stream.filter(Files::isDirectory).collect(Collectors.toList());
        }
        if (roots.size() != 1) {
            throw new IOException("Expected one extracted root directory but got " + roots.size());
        }
        return roots.get(0);
    }

    private static List<Path> jsonFilesByPrefix(Path parsingDir, String prefix) throws IOException {
        try (Stream<Path> stream = Files.list(parsingDir)) {
            return stream.filter(p -> p.getFileName().toString().startsWith(prefix))
                    .filter(p -> p.getFileName().toString().endsWith(".json"))
                    .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                    .collect(Collectors.toList());
        }
    }

    private static CorpusRunSummary runJsonTestSuiteParsingCorpus(Path parsingDir)
            throws IOException {
        List<Path> files;
        try (Stream<Path> stream = Files.list(parsingDir)) {
            files =
                    stream.filter(p -> p.getFileName().toString().endsWith(".json"))
                            .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                            .collect(Collectors.toList());
        }

        Assumptions.assumeFalse(files.isEmpty(), () -> "No test files found in: " + parsingDir);

        CorpusRunSummary summary = new CorpusRunSummary();
        for (Path file : files) {
            String name = file.getFileName().toString();
            if (name.length() < 2 || name.charAt(1) != '_') {
                continue;
            }

            char kind = Character.toLowerCase(name.charAt(0));
            if (kind != 'y' && kind != 'n' && kind != 'i') {
                continue;
            }

            byte[] bytes = Files.readAllBytes(file);
            String input;
            try {
                input = decodeStrictUtf8(bytes);
            } catch (CharacterCodingException e) {
                summary.onDecodeFailure(kind, name);
                continue;
            }

            boolean parsed;
            try {
                Json.parse(input);
                parsed = true;
            } catch (RuntimeException ex) {
                parsed = false;
            }

            summary.onParsed(kind, parsed, name);
        }
        return summary;
    }

    private static String decodeStrictUtf8(byte[] bytes) throws CharacterCodingException {
        return StandardCharsets.UTF_8
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString();
    }

    private static void deleteRecursivelyIfExists(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (Stream<Path> stream = Files.walk(root)) {
            List<Path> paths =
                    stream.sorted(Comparator.reverseOrder()).collect(Collectors.toList());
            for (Path path : paths) {
                Files.deleteIfExists(path);
            }
        }
    }

    private static final class CorpusRunSummary {
        private int yPassed;
        private int yFailed;
        private int nPassed;
        private int nFailed;
        private int iAccepted;
        private int iRejected;
        private int iUndecodable;
        private final StringBuilder failures = new StringBuilder();

        private void onDecodeFailure(char kind, String name) {
            if (kind == 'y') {
                yFailed++;
                failures.append("Y failed (UTF-8 decode): ").append(name).append('\n');
            } else if (kind == 'n') {
                nPassed++;
            } else {
                iUndecodable++;
            }
        }

        private void onParsed(char kind, boolean parsed, String name) {
            if (kind == 'y') {
                if (parsed) {
                    yPassed++;
                } else {
                    yFailed++;
                    failures.append("Y failed (rejected): ").append(name).append('\n');
                }
            } else if (kind == 'n') {
                if (!parsed) {
                    nPassed++;
                } else {
                    nFailed++;
                    failures.append("N failed (accepted): ").append(name).append('\n');
                }
            } else {
                if (parsed) {
                    iAccepted++;
                } else {
                    iRejected++;
                }
            }
        }

        private String toSummaryString() {
            return String.format(
                    Locale.ROOT,
                    "JSONTestSuite summary: y pass=%d fail=%d | n pass=%d fail=%d | i accept=%d"
                            + " reject=%d undecodable=%d",
                    yPassed,
                    yFailed,
                    nPassed,
                    nFailed,
                    iAccepted,
                    iRejected,
                    iUndecodable);
        }
    }
}

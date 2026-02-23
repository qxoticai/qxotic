package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Runs the upstream JSONTestSuite corpus (https://github.com/nst/JSONTestSuite).
 *
 * <p>Configuration (first non-empty wins):
 *
 * <ol>
 *   <li>System property: {@code -Djson.testsuite.path=/path/to/JSONTestSuite}
 *   <li>Environment variable: {@code JSON_TEST_SUITE_PATH}
 *   <li>Default: {@code /tmp/JSONTestSuite}
 * </ol>
 */
class JSONTestSuiteCorpusTest {

    private static final String PROP_PATH = "json.testsuite.path";
    private static final String ENV_PATH = "JSON_TEST_SUITE_PATH";
    private static final Path DEFAULT_PATH = Paths.get(System.getProperty("user.home"), ".cache", "qxotic", "json", "JSONTestSuite");

    @Test
    void runJsonTestSuiteCorpus() throws IOException {
        Path suiteRoot = resolveSuiteRoot();
        Path parsingDir = suiteRoot.resolve("test_parsing");

        if (!Files.isDirectory(parsingDir)) {
            System.err.println("\n" + "=".repeat(70));
            System.err.println("WARNING: JSONTestSuite corpus not found");
            System.err.println("=".repeat(70));
            System.err.println();
            System.err.println("The JSONTestSuite test corpus is required to run these tests.");
            System.err.println();
            System.err.println("To download it, run:");
            System.err.println();
            System.err.println("  mkdir -p ~/.cache/qxotic/json");
            System.err.println("  git clone https://github.com/nst/JSONTestSuite.git ~/.cache/qxotic/json/JSONTestSuite");
            System.err.println();
            System.err.println("Or set a custom path using:");
            System.err.println("  - System property: -Djson.testsuite.path=/path/to/JSONTestSuite");
            System.err.println("  - Environment variable: JSON_TEST_SUITE_PATH=/path/to/JSONTestSuite");
            System.err.println("=".repeat(70));
            System.err.println();
        }

        Assumptions.assumeTrue(
                Files.isDirectory(parsingDir),
                () -> "JSONTestSuite not found at: " + suiteRoot);

        List<Path> files;
        try (Stream<Path> stream = Files.list(parsingDir)) {
            files =
                    stream.filter(p -> p.getFileName().toString().endsWith(".json"))
                            .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                            .collect(Collectors.toList());
        }

        Assumptions.assumeFalse(files.isEmpty(), () -> "No test files found in: " + parsingDir);

        int yPassed = 0;
        int yFailed = 0;
        int nPassed = 0;
        int nFailed = 0;
        int iAccepted = 0;
        int iRejected = 0;
        int iUndecodable = 0;

        StringBuilder failures = new StringBuilder();

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
                if (kind == 'y') {
                    yFailed++;
                    failures.append("Y failed (UTF-8 decode): ").append(name).append('\n');
                } else if (kind == 'n') {
                    nPassed++;
                } else {
                    iUndecodable++;
                }
                continue;
            }

            boolean parsed;
            try {
                JSON.parse(input);
                parsed = true;
            } catch (RuntimeException ex) {
                parsed = false;
            }

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

        System.out.println(
                String.format(
                        Locale.ROOT,
                        "JSONTestSuite summary: y pass=%d fail=%d | n pass=%d fail=%d | i accept=%d reject=%d undecodable=%d",
                        yPassed,
                        yFailed,
                        nPassed,
                        nFailed,
                        iAccepted,
                        iRejected,
                        iUndecodable));

        assertTrue(yFailed == 0, () -> "Some y_ files were rejected:\n" + failures);
        assertFalse(nFailed > 0, () -> "Some n_ files were accepted:\n" + failures);
    }

    private static Path resolveSuiteRoot() {
        String fromProp = System.getProperty(PROP_PATH);
        if (fromProp != null && !fromProp.isBlank()) {
            return Paths.get(fromProp.trim());
        }

        String fromEnv = System.getenv(ENV_PATH);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return Paths.get(fromEnv.trim());
        }

        return DEFAULT_PATH;
    }

    private static String decodeStrictUtf8(byte[] bytes) throws CharacterCodingException {
        return StandardCharsets.UTF_8
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString();
    }
}

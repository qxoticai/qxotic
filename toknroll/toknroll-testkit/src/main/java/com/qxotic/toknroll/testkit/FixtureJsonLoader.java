package com.qxotic.toknroll.testkit;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Reads a JSON fixture from the test classpath, else from the repository's test-fixtures. */
public final class FixtureJsonLoader {

    private static final String GROUND_TRUTH_GENERATOR =
            "python3 toknroll-benchmarks/generate_ground_truth.py";

    private FixtureJsonLoader() {}

    static Map<String, Object> loadMap(Class<?> owner, String resourceName, String failureContext) {
        return loadMap(owner, resourceName, failureContext, GROUND_TRUTH_GENERATOR);
    }

    /**
     * The fixture's top-level object; a missing fixture fails with the command that generates it.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadMap(
            Class<?> owner, String resourceName, String failureContext, String generator) {
        try (InputStream is = owner.getClassLoader().getResourceAsStream(resourceName)) {
            if (is != null) {
                String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
                return (Map<String, Object>) Json.parse(json);
            }

            Path diskPath = firstExistingFixturePath(resourceName);
            if (Files.exists(diskPath)) {
                String json = Files.readString(diskPath, StandardCharsets.UTF_8);
                return (Map<String, Object>) Json.parse(json);
            }

            throw new IllegalStateException(
                    missingFixtureMessage(
                            resourceName, candidateFixturePaths(resourceName), generator));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load " + failureContext, e);
        }
    }

    private static Path firstExistingFixturePath(String resourceName) {
        for (Path candidate : candidateFixturePaths(resourceName)) {
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        return candidateFixturePaths(resourceName).get(0);
    }

    private static List<Path> candidateFixturePaths(String resourceName) {
        String rootOverride = System.getProperty("toknroll.test.fixtureDir");
        if (rootOverride != null && !rootOverride.isBlank()) {
            return List.of(Paths.get(rootOverride).resolve(resourceName));
        }

        List<Path> candidates = new ArrayList<>();
        Path gitRoot = GitRepoPaths.findGitRoot();
        if (gitRoot != null) {
            candidates.add(gitRoot.resolve("test-fixtures").resolve(resourceName));
        }

        candidates.add(Paths.get("..", "test-fixtures").resolve(resourceName));
        candidates.add(Paths.get("test-fixtures").resolve(resourceName));
        candidates.add(Paths.get("src", "test", "resources").resolve(resourceName));
        candidates.add(
                Paths.get("..", "toknroll-core", "src", "test", "resources").resolve(resourceName));
        candidates.add(
                Paths.get("toknroll-core", "src", "test", "resources").resolve(resourceName));
        return candidates;
    }

    private static String missingFixtureMessage(
            String resourceName, List<Path> checkedPaths, String generator) {
        return "Missing fixture "
                + resourceName
                + " (also checked "
                + checkedPaths
                + "). Generate it with: "
                + generator;
    }
}

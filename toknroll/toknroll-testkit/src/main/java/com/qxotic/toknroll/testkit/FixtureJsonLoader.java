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

final class FixtureJsonLoader {

    private FixtureJsonLoader() {}

    @SuppressWarnings("unchecked")
    static Map<String, Object> loadMap(Class<?> owner, String resourceName, String failureContext) {
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
                    missingFixtureMessage(resourceName, candidateFixturePaths(resourceName)));
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

    private static String missingFixtureMessage(String resourceName, List<Path> checkedPaths) {
        return "Missing fixture "
                + resourceName
                + " (also checked "
                + checkedPaths
                + "). Generate fixtures with: python3 benchmarks/generate_ground_truth.py";
    }
}

package com.qxotic.toknroll.testkit;

import java.nio.file.Files;
import java.nio.file.Path;

public final class GoldenFixturePaths {
    private GoldenFixturePaths() {}

    public static Path resolveGoldenEnwik8Dir() {
        String override = System.getProperty(TestSystemProperties.GOLDEN_DIR);
        if (override != null && !override.isBlank()) {
            return Path.of(override).toAbsolutePath().normalize();
        }

        Path cacheCandidate =
                TestCachePaths.resolveUnderTestArtifacts("golden", "enwik8")
                        .toAbsolutePath()
                        .normalize();
        if (Files.isDirectory(cacheCandidate)) {
            return cacheCandidate;
        }

        Path gitRoot = GitRepoPaths.findGitRoot();
        if (gitRoot != null) {
            Path repoCandidate =
                    gitRoot.resolve("test-fixtures").resolve("golden").resolve("enwik8");
            if (Files.isDirectory(repoCandidate)) {
                return repoCandidate.toAbsolutePath().normalize();
            }

            Path nestedRepoCandidate =
                    gitRoot.resolve("toknroll")
                            .resolve("test-fixtures")
                            .resolve("golden")
                            .resolve("enwik8");
            if (Files.isDirectory(nestedRepoCandidate)) {
                return nestedRepoCandidate.toAbsolutePath().normalize();
            }
        }

        Path parentCandidate =
                Path.of("..", "test-fixtures", "golden", "enwik8").toAbsolutePath().normalize();
        if (Files.isDirectory(parentCandidate)) {
            return parentCandidate;
        }

        return Path.of("test-fixtures", "golden", "enwik8").toAbsolutePath().normalize();
    }
}

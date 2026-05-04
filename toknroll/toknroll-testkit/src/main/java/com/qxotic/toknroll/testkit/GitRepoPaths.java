package com.qxotic.toknroll.testkit;

import java.nio.file.Files;
import java.nio.file.Path;

public final class GitRepoPaths {
    private GitRepoPaths() {}

    public static Path findGitRoot() {
        return findGitRoot(Path.of("").toAbsolutePath().normalize());
    }

    public static Path findGitRoot(Path start) {
        Path current = start;
        while (current != null) {
            if (Files.exists(current.resolve(".git"))) {
                return current;
            }
            current = current.getParent();
        }
        return null;
    }
}

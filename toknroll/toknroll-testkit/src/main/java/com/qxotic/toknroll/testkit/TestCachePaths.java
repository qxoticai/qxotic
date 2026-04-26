package com.qxotic.toknroll.testkit;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/** OS-aware cache path resolver for test artifacts. */
public final class TestCachePaths {

    public static final String CACHE_ROOT_PROPERTY = "toknroll.test.cacheRoot";

    private static final String ORG_DIR = "qxotic";
    private static final String APP_DIR = "toknroll";
    private static final String TEST_ARTIFACTS_DIR = "test-artifacts";

    private TestCachePaths() {}

    public static Path testArtifactsRoot() {
        String override = System.getProperty(CACHE_ROOT_PROPERTY);
        if (override != null && !override.isBlank()) {
            return Paths.get(override);
        }

        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String home = System.getProperty("user.home", ".");

        if (os.contains("win")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isBlank()) {
                return Paths.get(localAppData, ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
            }
            String appData = System.getenv("APPDATA");
            if (appData != null && !appData.isBlank()) {
                return Paths.get(appData, ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
            }
            return Paths.get(home, "AppData", "Local", ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
        }

        if (os.contains("mac") || os.contains("darwin")) {
            return Paths.get(home, "Library", "Caches", ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
        }

        String xdgCacheHome = System.getenv("XDG_CACHE_HOME");
        if (xdgCacheHome != null && !xdgCacheHome.isBlank()) {
            return Paths.get(xdgCacheHome, ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
        }
        return Paths.get(home, ".cache", ORG_DIR, APP_DIR, TEST_ARTIFACTS_DIR);
    }

    public static Path resolveUnderTestArtifacts(String first, String... more) {
        Path path = testArtifactsRoot().resolve(first);
        if (more != null) {
            for (String part : more) {
                path = path.resolve(part);
            }
        }
        return path;
    }
}

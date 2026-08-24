package com.qxotic.toknroll.testkit;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/** OS-aware cache path resolver for test artifacts and downloaded fixtures. */
public final class TestCachePaths {

    /** Overrides the test-artifacts directory directly (legacy, most specific). */
    public static final String CACHE_ROOT_PROPERTY = "toknroll.test.cacheRoot";

    /** Overrides the toknroll cache root; mirrors the runtime readers in the HF/GGUF modules. */
    public static final String ROOT_PROPERTY = TestSystemProperties.ARTIFACT_CACHE_ROOT;

    public static final String ROOT_ENV = "TOKNROLL_CACHE_ROOT";

    private static final String ORG_DIR = "qxotic";
    private static final String APP_DIR = "toknroll";
    private static final String TEST_ARTIFACTS_DIR = "test-artifacts";

    private TestCachePaths() {}

    /**
     * The OS-aware toknroll cache root (override: -Dtoknroll.cache.root or TOKNROLL_CACHE_ROOT).
     */
    public static Path cacheRoot() {
        String override = System.getProperty(ROOT_PROPERTY);
        if (override == null || override.isBlank()) {
            override = System.getenv(ROOT_ENV);
        }
        if (override != null && !override.isBlank()) {
            return Paths.get(override);
        }
        return osCacheDir().resolve(ORG_DIR).resolve(APP_DIR);
    }

    public static Path testArtifactsRoot() {
        String override = System.getProperty(CACHE_ROOT_PROPERTY);
        if (override != null && !override.isBlank()) {
            return Paths.get(override);
        }
        return cacheRoot().resolve(TEST_ARTIFACTS_DIR);
    }

    /**
     * The shared corpus directory: expensive downloads (enwik8/enwik9) live here, OUTSIDE
     * test-artifacts, so wiping disposable test output never forces a re-download.
     */
    public static Path corpusDir() {
        return cacheRoot().resolve("corpus");
    }

    private static Path osCacheDir() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String home = System.getProperty("user.home", ".");

        if (os.contains("win")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isBlank()) {
                return Paths.get(localAppData);
            }
            String appData = System.getenv("APPDATA");
            if (appData != null && !appData.isBlank()) {
                return Paths.get(appData);
            }
            return Paths.get(home, "AppData", "Local");
        }

        if (os.contains("mac") || os.contains("darwin")) {
            return Paths.get(home, "Library", "Caches");
        }

        String xdgCacheHome = System.getenv("XDG_CACHE_HOME");
        if (xdgCacheHome != null && !xdgCacheHome.isBlank()) {
            return Paths.get(xdgCacheHome);
        }
        return Paths.get(home, ".cache");
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

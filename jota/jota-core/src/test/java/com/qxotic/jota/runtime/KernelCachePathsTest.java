package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.jota.Device;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class KernelCachePathsTest {

    private static final String CACHE_ROOT_PROPERTY = KernelCachePaths.CACHE_ROOT_PROPERTY;
    private static final String VERSION_PROPERTY = KernelCachePaths.VERSION_PROPERTY;
    private static final String ORIGINAL_CACHE_ROOT = System.getProperty(CACHE_ROOT_PROPERTY);
    private static final String ORIGINAL_VERSION = System.getProperty(VERSION_PROPERTY);

    @AfterEach
    void restoreProperties() {
        restore(CACHE_ROOT_PROPERTY, ORIGINAL_CACHE_ROOT);
        restore(VERSION_PROPERTY, ORIGINAL_VERSION);
    }

    @Test
    void usesConfiguredRootAndVersionWhenProvided() {
        System.setProperty(CACHE_ROOT_PROPERTY, "/tmp/jota-test-cache");
        System.setProperty(VERSION_PROPERTY, "v1.2.3");

        assertEquals(Path.of("/tmp/jota-test-cache"), KernelCachePaths.cacheRoot());
        assertEquals("v1.2.3", KernelCachePaths.version());
        assertEquals(
                Path.of("/tmp/jota-test-cache", "v1.2.3", Device.PANAMA.leafName()),
                KernelCachePaths.deviceRoot(Device.PANAMA));
        assertEquals(
                Path.of("/tmp/jota-test-cache", "v1.2.3", Device.HIP.leafName(), "programs"),
                KernelCachePaths.programRoot(Device.HIP));
    }

    @Test
    void sanitizesVersionSegment() {
        System.setProperty(VERSION_PROPERTY, "release/1:2 3");

        assertEquals("release_1_2_3", KernelCachePaths.version());
    }

    @Test
    void resolvesLinuxDefaultCacheRoot() {
        String xdgCacheHome = System.getenv("XDG_CACHE_HOME");
        Path expected =
                (xdgCacheHome != null && !xdgCacheHome.isBlank())
                        ? Path.of(xdgCacheHome).resolve("jota")
                        : Path.of("/home/tester").resolve(".cache").resolve("jota");

        assertEquals(
                expected,
                KernelCachePaths.defaultCacheRootFor("Linux", "/home/tester", xdgCacheHome, null));
    }

    @Test
    void resolvesMacDefaultCacheRoot() {
        assertEquals(
                Path.of("/Users/tester", "Library", "Caches", "jota"),
                KernelCachePaths.defaultCacheRootFor("Mac OS X", "/Users/tester", null, null));
    }

    @Test
    void resolvesWindowsDefaultCacheRoot() {
        String localAppData = System.getenv("LOCALAPPDATA");
        Path expected =
                (localAppData != null && !localAppData.isBlank())
                        ? Path.of(localAppData).resolve("jota").resolve("cache")
                        : Path.of("C:\\Users\\tester")
                                .resolve("AppData")
                                .resolve("Local")
                                .resolve("jota")
                                .resolve("cache");

        assertEquals(
                expected,
                KernelCachePaths.defaultCacheRootFor(
                        "Windows 11", "C:\\Users\\tester", null, localAppData));
    }

    @Test
    void defaultVersionIsFilesystemSafe() {
        String version = KernelCachePaths.version();

        assertFalse(version.isBlank());
        assertFalse(version.contains("/"));
        assertFalse(version.contains("\\"));
        assertEquals(version, version.replaceAll("[^A-Za-z0-9._-]", "_"));
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
            return;
        }
        System.setProperty(key, value);
    }
}

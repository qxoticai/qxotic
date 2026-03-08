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

    @AfterEach
    void restoreProperties() {
        clear(CACHE_ROOT_PROPERTY);
        clear(VERSION_PROPERTY);
        clear("os.name");
        clear("user.home");
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
        System.setProperty("os.name", "Linux");
        System.setProperty("user.home", "/home/tester");

        String xdgCacheHome = System.getenv("XDG_CACHE_HOME");
        Path expected =
                (xdgCacheHome != null && !xdgCacheHome.isBlank())
                        ? Path.of(xdgCacheHome).resolve("jota")
                        : Path.of("/home/tester").resolve(".cache").resolve("jota");

        assertEquals(expected, KernelCachePaths.cacheRoot());
    }

    @Test
    void resolvesMacDefaultCacheRoot() {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("user.home", "/Users/tester");

        assertEquals(
                Path.of("/Users/tester", "Library", "Caches", "jota"),
                KernelCachePaths.cacheRoot());
    }

    @Test
    void resolvesWindowsDefaultCacheRoot() {
        System.setProperty("os.name", "Windows 11");
        System.setProperty("user.home", "C:\\Users\\tester");

        String localAppData = System.getenv("LOCALAPPDATA");
        Path expected =
                (localAppData != null && !localAppData.isBlank())
                        ? Path.of(localAppData).resolve("jota").resolve("cache")
                        : Path.of("C:\\Users\\tester")
                                .resolve("AppData")
                                .resolve("Local")
                                .resolve("jota")
                                .resolve("cache");

        assertEquals(expected, KernelCachePaths.cacheRoot());
    }

    @Test
    void defaultVersionIsFilesystemSafe() {
        String version = KernelCachePaths.version();

        assertFalse(version.isBlank());
        assertFalse(version.contains("/"));
        assertFalse(version.contains("\\"));
        assertEquals(version, version.replaceAll("[^A-Za-z0-9._-]", "_"));
    }

    private static void clear(String key) {
        System.clearProperty(key);
    }
}

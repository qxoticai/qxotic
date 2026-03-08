package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.nio.file.Path;
import java.util.Set;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

public final class KernelCachePaths {

    public static final String CACHE_ROOT_PROPERTY = "jota.cache.root";
    public static final String VERSION_PROPERTY = "jota.version";
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");
    private static final AtomicBoolean CONFIG_LOGGED = new AtomicBoolean(false);
    private static final Set<Device> DEVICE_ROOT_LOGGED = ConcurrentHashMap.newKeySet();

    private KernelCachePaths() {}

    public static Path cacheRoot() {
        String configured = System.getProperty(CACHE_ROOT_PROPERTY);
        Path root;
        if (configured != null && !configured.isBlank()) {
            root = Path.of(configured.trim());
        } else {
            root = defaultCacheRoot();
        }
        logConfigurationOnce(root, version());
        return root;
    }

    public static String version() {
        String configured = System.getProperty(VERSION_PROPERTY);
        if (configured != null && !configured.isBlank()) {
            return sanitizeSegment(configured.trim());
        }
        Package pkg = Device.class.getPackage();
        String implementationVersion = pkg == null ? null : pkg.getImplementationVersion();
        if (implementationVersion != null && !implementationVersion.isBlank()) {
            return sanitizeSegment(implementationVersion.trim());
        }
        return "dev";
    }

    public static Path versionRoot() {
        return cacheRoot().resolve(version());
    }

    public static Path deviceRoot(Device device) {
        Path root = versionRoot().resolve(device.leafName());
        logDeviceRootOnce(device, root);
        return root;
    }

    public static Path programRoot(Device device) {
        return deviceRoot(device).resolve("programs");
    }

    private static Path defaultCacheRoot() {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String userHome = System.getProperty("user.home");

        if (osName.contains("win")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isBlank()) {
                return Path.of(localAppData).resolve("jota").resolve("cache");
            }
            if (userHome != null && !userHome.isBlank()) {
                return Path.of(userHome)
                        .resolve("AppData")
                        .resolve("Local")
                        .resolve("jota")
                        .resolve("cache");
            }
        }

        if (osName.contains("mac") || osName.contains("darwin")) {
            if (userHome != null && !userHome.isBlank()) {
                return Path.of(userHome).resolve("Library").resolve("Caches").resolve("jota");
            }
        }

        String xdgCacheHome = System.getenv("XDG_CACHE_HOME");
        if (xdgCacheHome != null && !xdgCacheHome.isBlank()) {
            return Path.of(xdgCacheHome).resolve("jota");
        }
        if (userHome != null && !userHome.isBlank()) {
            return Path.of(userHome).resolve(".cache").resolve("jota");
        }

        return Path.of(System.getProperty("java.io.tmpdir", ".")).resolve("jota-cache");
    }

    private static String sanitizeSegment(String value) {
        return value.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static void logConfigurationOnce(Path root, String version) {
        if (KERNEL_LOG && CONFIG_LOGGED.compareAndSet(false, true)) {
            System.out.println(
                    "[jota-kernel] cacheRoot=" + root.toAbsolutePath() + " version=" + version);
        }
    }

    private static void logDeviceRootOnce(Device device, Path root) {
        if (KERNEL_LOG && DEVICE_ROOT_LOGGED.add(device)) {
            System.out.println(
                    "[jota-kernel] device="
                            + device.leafName()
                            + " deviceRoot="
                            + root.toAbsolutePath());
        }
    }
}

package com.qxotic.jota.runtime;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Utility for loading native libraries from JAR resources or system path.
 *
 * <p>Loading order:
 *
 * <ol>
 *   <li>Try system library path (allows user override via -Djava.library.path)
 *   <li>Extract from JAR to temp directory and load
 * </ol>
 *
 * <p>Supported platforms:
 *
 * <ul>
 *   <li>Linux x86_64
 *   <li>Linux aarch64
 *   <li>Windows x86_64
 *   <li>macOS aarch64 (Apple Silicon)
 * </ul>
 */
public final class NativeLibraryLoader {

    private static final String NATIVE_RESOURCE_PREFIX = "/META-INF/native";
    private static final Set<String> LOADED_LIBRARIES = ConcurrentHashMap.newKeySet();

    // Supported platform combinations
    private static final Set<String> SUPPORTED_PLATFORMS =
            Set.of("linux-x86_64", "linux-aarch64", "windows-x86_64", "macos-aarch64");

    private NativeLibraryLoader() {
        // Utility class
    }

    /**
     * Loads a native library by name.
     *
     * @param libraryName the library name without prefix/suffix (e.g., "jota_c")
     * @throws UnsatisfiedLinkError if the library cannot be loaded
     */
    public static void load(String libraryName) {
        if (LOADED_LIBRARIES.contains(libraryName)) {
            return;
        }

        synchronized (NativeLibraryLoader.class) {
            if (LOADED_LIBRARIES.contains(libraryName)) {
                return;
            }

            try {
                // 1. Try system library path first (user override)
                System.loadLibrary(libraryName);
                LOADED_LIBRARIES.add(libraryName);
                return;
            } catch (UnsatisfiedLinkError e) {
                // Not in java.library.path, try JAR
            }

            // 2. Extract from JAR and load
            loadFromJar(libraryName);
            LOADED_LIBRARIES.add(libraryName);
        }
    }

    private static void loadFromJar(String libraryName) {
        Platform platform = Platform.detect();
        String libFileName = System.mapLibraryName(libraryName);

        // Check platform support
        String platformKey = platform.os() + "-" + platform.arch();
        if (!SUPPORTED_PLATFORMS.contains(platformKey)) {
            throw new UnsatisfiedLinkError(
                    String.format(
                            "Platform not supported: %s (%s). Supported platforms: Linux"
                                + " x86_64/aarch64, Windows x86_64, macOS aarch64 (Apple Silicon)."
                                + " You can provide your own native library via"
                                + " -Djava.library.path",
                            platform.os(), platform.arch()));
        }

        String resourcePath =
                String.format(
                        "%s/%s/%s/%s",
                        NATIVE_RESOURCE_PREFIX, platform.os(), platform.arch(), libFileName);

        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new UnsatisfiedLinkError(
                        String.format(
                                "Native library not found in JAR: %s. Platform: %s %s. Either the"
                                    + " backend JAR doesn't include natives for this platform, or"
                                    + " you need to provide the library via -Djava.library.path",
                                resourcePath, platform.os(), platform.arch()));
            }

            // Create temp directory
            Path tempDir = createTempDirectory(libraryName);
            Path tempFile = tempDir.resolve(libFileName);

            try {
                // Extract
                Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);

                // Make executable on Unix
                if (!platform.isWindows()) {
                    tempFile.toFile().setExecutable(true, false);
                }

                // Load
                System.load(tempFile.toAbsolutePath().toString());

            } catch (IOException e) {
                throw new UnsatisfiedLinkError(
                        "Failed to extract native library: "
                                + libraryName
                                + " - "
                                + e.getMessage());
            }

        } catch (IOException e) {
            throw new UnsatisfiedLinkError(
                    "Failed to read native library from JAR: "
                            + libraryName
                            + " - "
                            + e.getMessage());
        }
    }

    private static Path createTempDirectory(String libraryName) throws IOException {
        // Create a unique temp directory for this library
        String tempDirName = "jota-native-" + libraryName + "-" + System.nanoTime();
        Path tempDir = Files.createTempDirectory(tempDirName);

        // Schedule cleanup on JVM exit
        tempDir.toFile().deleteOnExit();

        return tempDir;
    }

    /** Returns true if the given platform is supported. */
    public static boolean isPlatformSupported() {
        Platform platform = Platform.detect();
        String platformKey = platform.os() + "-" + platform.arch();
        return SUPPORTED_PLATFORMS.contains(platformKey);
    }

    /** Returns the current platform string (e.g., "linux-x86_64"). */
    public static String currentPlatform() {
        Platform platform = Platform.detect();
        return platform.os() + "-" + platform.arch();
    }

    /** Platform information. */
    private record Platform(String os, String arch) {

        static Platform detect() {
            String osName = System.getProperty("os.name", "unknown").toLowerCase(Locale.ROOT);
            String osArch = System.getProperty("os.arch", "unknown").toLowerCase(Locale.ROOT);

            // Detect OS
            String os;
            if (osName.contains("linux")) {
                os = "linux";
            } else if (osName.contains("mac") || osName.contains("darwin")) {
                os = "macos";
            } else if (osName.contains("win")) {
                os = "windows";
            } else {
                os = osName;
            }

            // Detect architecture (normalize to standard names)
            String arch;
            if (osArch.equals("amd64") || osArch.equals("x86_64")) {
                arch = "x86_64";
            } else if (osArch.equals("aarch64") || osArch.equals("arm64")) {
                arch = "aarch64";
            } else {
                arch = osArch;
            }

            return new Platform(os, arch);
        }

        boolean isWindows() {
            return os.equals("windows");
        }
    }
}

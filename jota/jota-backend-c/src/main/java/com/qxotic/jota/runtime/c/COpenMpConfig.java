package com.qxotic.jota.runtime.c;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

final class COpenMpConfig {
    private COpenMpConfig() {}

    static boolean enabled() {
        String override = System.getProperty("jota.c.openmp");
        if (override != null) {
            return Boolean.parseBoolean(override);
        }
        if (isWindows()) {
            return false;
        }
        if (!isMac()) {
            return true;
        }
        return detectBrewLibOmpPrefix() != null;
    }

    static boolean isMac() {
        return osName().contains("mac");
    }

    static boolean isWindows() {
        return osName().contains("win");
    }

    private static String osName() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    }

    static Path detectBrewLibOmpPrefix() {
        Path homebrew = Path.of("/opt/homebrew/opt/libomp");
        if (Files.isDirectory(homebrew)) {
            return homebrew;
        }
        Path intel = Path.of("/usr/local/opt/libomp");
        if (Files.isDirectory(intel)) {
            return intel;
        }
        return null;
    }
}

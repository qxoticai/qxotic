package com.qxotic.jam.libjam;

import com.qxotic.jam.JAM;

/**
 * Provider for the native {@code libjam} backend (id {@code native}). Highest priority (1000);
 * available when the bundled library for the current OS/arch loads and exports the expected
 * symbols. Skipped silently when there is no library for this platform, with a warning when there
 * is one that does not load.
 */
public final class NativeJAMProvider implements JAM.Provider {
    @Override
    public String id() {
        return "native";
    }

    @Override
    public int priority() {
        return 1000;
    }

    @Override
    public boolean isAvailable() {
        try {
            NativeLoader.load();
        } catch (Throwable t) {
            // No library for this platform is the expected, silent fallback. One that exists but
            // does not load, the user's or the bundled one, says so.
            if (!NativeLoader.override().isEmpty() || NativeLoader.bundled())
                warn(
                        "jam: the native library did not load, using the Java backends: "
                                + t
                                + " (a temporary directory mounted noexec? set -Djava.io.tmpdir)");
            return false;
        }
        try {
            NativeJAM.probe();
            return true;
        } catch (Throwable t) {
            // The library loaded but does not export what this build binds: a stale or partial
            // bundle. Falling back is right for the caller, but never silently.
            warn("jam: bundled native library is unusable, using the Java backends: " + t);
            return false;
        }
    }

    private static void warn(String message) {
        System.getLogger(NativeJAMProvider.class.getName())
                .log(System.Logger.Level.WARNING, message);
    }

    @Override
    public JAM create(JAM.Parallel parallel) {
        return NativeJAM.create(parallel);
    }
}

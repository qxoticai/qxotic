package com.qxotic.jam.libjam;

import com.qxotic.jam.JAM;

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
            return false; // no library for this platform: the expected, silent fallback
        }
        try {
            NativeJAM.probe();
            return true;
        } catch (Throwable t) {
            // The library loaded but does not export what this build binds: a stale or partial
            // bundle. Falling back is right for the caller, but never silently.
            System.getLogger(NativeJAMProvider.class.getName())
                    .log(
                            System.Logger.Level.WARNING,
                            "jam: bundled native library is unusable, using the Java backends: "
                                    + t);
            return false;
        }
    }

    @Override
    public JAM create(JAM.Parallel parallel) {
        return NativeJAM.create(parallel);
    }
}

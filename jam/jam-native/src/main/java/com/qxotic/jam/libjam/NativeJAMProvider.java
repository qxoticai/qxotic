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
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    @Override
    public JAM create(JAM.Parallel parallel) {
        NativeJAM.host = parallel;
        return NativeJAM.global();
    }
}

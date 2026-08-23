package com.qxotic.jam;

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
    public JAM create() {
        return NativeJAM.global();
    }
}

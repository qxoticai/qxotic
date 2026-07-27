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
            NativeJAM.global();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    @Override
    public JAM create() {
        // NativeJAM is a single shared native context today; concurrent mm calls may return EBUSY.
        return NativeJAM.global();
    }
}

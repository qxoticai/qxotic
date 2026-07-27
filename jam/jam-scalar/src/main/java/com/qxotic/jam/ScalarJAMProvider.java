package com.qxotic.jam;

public final class ScalarJAMProvider implements JAM.Provider {
    @Override
    public String id() {
        return "scalar";
    }

    @Override
    public int priority() {
        return 0;
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public JAM create() {
        return new ScalarJAM();
    }
}

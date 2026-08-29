package com.qxotic.jam.scalar;

import com.qxotic.jam.JAM;

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
    public JAM create(JAM.Parallel parallel) {
        return new ScalarJAM(parallel);
    }
}

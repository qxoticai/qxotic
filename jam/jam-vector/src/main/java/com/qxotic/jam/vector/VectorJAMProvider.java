package com.qxotic.jam.vector;

import com.qxotic.jam.JAM;

public final class VectorJAMProvider implements JAM.Provider {
    @Override
    public String id() {
        return "vector";
    }

    @Override
    public int priority() {
        return 500;
    }

    @Override
    public boolean isAvailable() {
        return VectorJAM.isAvailable();
    }

    @Override
    public JAM create(JAM.Parallel parallel) {
        return new VectorJAM(parallel);
    }
}

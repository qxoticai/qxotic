package com.qxotic.jam.vector;

import com.qxotic.jam.JAM;

/**
 * Provider for the Java Vector API backend (id {@code vector}). Available when {@code
 * jdk.incubator.vector} is on the module path; priority 500, below {@code native}.
 */
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

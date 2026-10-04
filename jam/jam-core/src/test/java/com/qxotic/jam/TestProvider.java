package com.qxotic.jam;

/**
 * A test-only provider so {@link JAMProvidersTest} exercises the real {@code ServiceLoader} path.
 */
public final class TestProvider implements JAM.Provider {

    static int availabilityChecks;
    static RuntimeException probeFailure;

    @Override
    public String id() {
        return "test";
    }

    @Override
    public int priority() {
        return Integer.MIN_VALUE;
    }

    @Override
    public boolean isAvailable() {
        availabilityChecks++;
        if (probeFailure != null) {
            throw probeFailure;
        }
        return true;
    }

    @Override
    public JAM create(JAM.Parallel parallel) {
        throw new UnsupportedOperationException("test provider never creates a backend");
    }
}

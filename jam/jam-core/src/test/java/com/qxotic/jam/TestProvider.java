package com.qxotic.jam;

/**
 * A test-only provider so {@link JAMProvidersTest} exercises the real {@code ServiceLoader} path.
 */
public final class TestProvider implements JAM.Provider {

    static int availabilityChecks;

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
        return true;
    }

    @Override
    public JAM create() {
        throw new UnsupportedOperationException("test provider never creates a backend");
    }
}

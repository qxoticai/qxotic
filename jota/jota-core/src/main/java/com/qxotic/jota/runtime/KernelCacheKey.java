package com.qxotic.jota.runtime;

import java.util.Objects;

public final class KernelCacheKey {

    private final String value;

    private KernelCacheKey(String value) {
        this.value = Objects.requireNonNull(value, "value");
    }

    public static KernelCacheKey of(String value) {
        return new KernelCacheKey(value);
    }

    public String value() {
        return value;
    }

    @Override
    public String toString() {
        return value;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelCacheKey that && value.equals(that.value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}

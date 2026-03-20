package com.qxotic.jota;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

public final class DeviceType {
    private static final Map<String, DeviceType> KNOWN = new ConcurrentHashMap<>();

    public static final DeviceType PANAMA = of("panama");
    public static final DeviceType JAVA = of("java");
    public static final DeviceType C = of("c");
    public static final DeviceType CUDA = of("cuda");
    public static final DeviceType HIP = of("hip");
    public static final DeviceType MOJO = of("mojo");
    public static final DeviceType OPENCL = of("opencl");
    public static final DeviceType METAL = of("metal");
    public static final DeviceType WEBGPU = of("webgpu");

    private final String id;

    private DeviceType(String id) {
        this.id = id;
    }

    static DeviceType of(String id) {
        String normalized = normalizeId(id);
        return KNOWN.computeIfAbsent(normalized, DeviceType::new);
    }

    public static DeviceType fromId(String id) {
        return of(id);
    }

    public String id() {
        return id;
    }

    public Device deviceIndex(int index) {
        return new Device(this, index);
    }

    private static String normalizeId(String id) {
        Objects.requireNonNull(id, "id");
        String normalized = id.strip().toLowerCase(Locale.ROOT);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        return normalized;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof DeviceType that)) {
            return false;
        }
        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return id;
    }
}

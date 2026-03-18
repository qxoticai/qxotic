package com.qxotic.jota;

import java.util.Locale;

public record DeviceType(String id) {
    public DeviceType {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("id must not be blank");
        id = id.strip().toLowerCase(Locale.ROOT);
    }

    public static final DeviceType PANAMA = new DeviceType("panama");
    public static final DeviceType JAVA = new DeviceType("java");
    public static final DeviceType C = new DeviceType("c");
    public static final DeviceType CUDA = new DeviceType("cuda");
    public static final DeviceType HIP = new DeviceType("hip");
    public static final DeviceType MOJO = new DeviceType("mojo");
    public static final DeviceType OPENCL = new DeviceType("opencl");
    public static final DeviceType METAL = new DeviceType("metal");
    public static final DeviceType WEBGPU = new DeviceType("webgpu");

    @Override
    public String toString() {
        return id;
    }
}

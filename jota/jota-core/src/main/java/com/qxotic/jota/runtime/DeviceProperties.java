package com.qxotic.jota.runtime;

import java.util.Map;
import java.util.OptionalLong;
import java.util.TreeMap;

public final class DeviceProperties {

    public static final DeviceProperties EMPTY = new DeviceProperties(Map.of());

    private final Map<String, Object> properties;

    public DeviceProperties(Map<String, Object> properties) {
        this.properties = Map.copyOf(properties);
    }

    public String getString(String key) {
        Object v = properties.get(key);
        if (v == null) {
            throw new IllegalArgumentException("No property: " + key);
        }
        return v.toString();
    }

    public long getLong(String key) {
        Object v = properties.get(key);
        if (v == null) {
            throw new IllegalArgumentException("No property: " + key);
        }
        if (v instanceof Number n) {
            return n.longValue();
        }
        throw new IllegalArgumentException("Property " + key + " is not a number: " + v);
    }

    public boolean getBoolean(String key) {
        Object v = properties.get(key);
        if (v == null) {
            throw new IllegalArgumentException("No property: " + key);
        }
        if (v instanceof Boolean b) {
            return b;
        }
        throw new IllegalArgumentException("Property " + key + " is not a boolean: " + v);
    }

    public OptionalLong optionalLong(String key) {
        Object v = properties.get(key);
        if (v == null) {
            return OptionalLong.empty();
        }
        if (v instanceof Number n) {
            return OptionalLong.of(n.longValue());
        }
        return OptionalLong.empty();
    }

    public boolean has(String key) {
        return properties.containsKey(key);
    }

    public String name() {
        return getString(DEVICE_NAME);
    }

    public String vendor() {
        return getString(VENDOR);
    }

    public String architecture() {
        return getString(ARCHITECTURE);
    }

    public long globalMemoryBytes() {
        return getLong(GLOBAL_MEMORY_BYTES);
    }

    public Map<String, Object> asMap() {
        return properties;
    }

    @Override
    public String toString() {
        if (properties.isEmpty()) {
            return "DeviceProperties{}";
        }
        var sb = new StringBuilder("DeviceProperties{\n");
        new TreeMap<>(properties)
                .forEach((k, v) -> sb.append("  ").append(k).append(" = ").append(v).append('\n'));
        sb.append('}');
        return sb.toString();
    }

    // --- Standard property keys ---

    // Identity
    public static final String DEVICE_NAME = "device.name";
    public static final String VENDOR = "device.vendor";
    public static final String ARCHITECTURE = "device.architecture";
    public static final String DRIVER_VERSION = "device.driver.version";

    // Memory
    public static final String GLOBAL_MEMORY_BYTES = "memory.global.bytes";
    public static final String SHARED_MEMORY_BYTES = "memory.shared.bytes";
    public static final String MAX_ALLOCATION_BYTES = "memory.max.allocation.bytes";
    public static final String L2_CACHE_BYTES = "memory.l2.cache.bytes";
    public static final String MEMORY_BUS_WIDTH_BITS = "memory.bus.width.bits";
    public static final String MEMORY_CLOCK_MHZ = "memory.clock.mhz";

    // Compute
    public static final String COMPUTE_UNITS = "compute.units";
    public static final String CLOCK_MHZ = "compute.clock.mhz";
    public static final String WARP_SIZE = "compute.warp.size";
    public static final String MAX_THREADS_PER_BLOCK = "compute.max.threads.per.block";
    public static final String MAX_BLOCK_DIM_X = "compute.max.block.dim.x";
    public static final String MAX_BLOCK_DIM_Y = "compute.max.block.dim.y";
    public static final String MAX_BLOCK_DIM_Z = "compute.max.block.dim.z";
    public static final String MAX_GRID_DIM_X = "compute.max.grid.dim.x";
    public static final String MAX_GRID_DIM_Y = "compute.max.grid.dim.y";
    public static final String MAX_GRID_DIM_Z = "compute.max.grid.dim.z";
    public static final String MAX_REGISTERS_PER_BLOCK = "compute.max.registers.per.block";
    public static final String MAX_SHARED_MEMORY_PER_BLOCK = "compute.max.shared.memory.per.block";
}

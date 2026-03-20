package com.qxotic.jota;

import java.util.Objects;

public final class Device {
    private final DeviceType type;
    private final long index;

    Device(DeviceType type, long index) {
        this.type = Objects.requireNonNull(type, "type");
        if (index < 0) {
            throw new IllegalArgumentException("index must be >= 0");
        }
        this.index = index;
    }

    public DeviceType type() {
        return type;
    }

    public long index() {
        return index;
    }

    public boolean belongsTo(DeviceType type) {
        return this.type.equals(type);
    }

    public String runtimeId() {
        return type.id();
    }

    public static Device defaultDevice() {
        return Environment.current().defaultDevice();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof Device device)) {
            return false;
        }
        return index == device.index && type.equals(device.type);
    }

    @Override
    public int hashCode() {
        return 31 * type.hashCode() + (int) index;
    }

    @Override
    public String toString() {
        return type.id() + ":" + index;
    }
}

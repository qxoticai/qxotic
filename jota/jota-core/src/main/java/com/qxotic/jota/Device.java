package com.qxotic.jota;

import java.util.Objects;

public record Device(DeviceType type, long index) {
    public Device(DeviceType type, long index) {
        this.type = Objects.requireNonNull(type, "type");
        if (index < 0) {
            throw new IllegalArgumentException("index must be >= 0");
        }
        this.index = index;
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
    public String toString() {
        return type.id() + ":" + index;
    }
}

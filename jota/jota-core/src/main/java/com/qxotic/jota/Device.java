package com.qxotic.jota;

import java.util.Objects;

public record Device(DeviceType type, int index) {
    public Device {
        Objects.requireNonNull(type, "type");
        if (index < 0) throw new IllegalArgumentException("index must be >= 0");
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
    public String toString() {
        return type.id() + ":" + index;
    }
}

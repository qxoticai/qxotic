package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import java.util.Optional;

public final class TestBackendContext {

    private static final ThreadLocal<Device> CURRENT = new ThreadLocal<>();

    private TestBackendContext() {}

    static void set(Device device) {
        CURRENT.set(device);
    }

    static void clear() {
        CURRENT.remove();
    }

    public static Optional<Device> current() {
        return Optional.ofNullable(CURRENT.get());
    }
}

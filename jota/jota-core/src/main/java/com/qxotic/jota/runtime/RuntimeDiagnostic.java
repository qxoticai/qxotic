package com.qxotic.jota.runtime;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.Objects;

public record RuntimeDiagnostic(String providerId, DeviceType deviceType, RuntimeProbe probe) {

    public RuntimeDiagnostic {
        providerId = Objects.requireNonNull(providerId, "providerId");
        deviceType = Objects.requireNonNull(deviceType, "deviceType");
        probe = Objects.requireNonNull(probe, "probe");
    }
}

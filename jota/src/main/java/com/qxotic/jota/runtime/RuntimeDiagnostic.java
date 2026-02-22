package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.Objects;

public record RuntimeDiagnostic(String providerId, Device device, RuntimeProbe probe) {

    public RuntimeDiagnostic {
        providerId = Objects.requireNonNull(providerId, "providerId");
        device = Objects.requireNonNull(device, "device");
        probe = Objects.requireNonNull(probe, "probe");
    }
}

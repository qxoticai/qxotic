package ai.qxotic.jota.runtime;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.Objects;

public record RuntimeDiagnostic(String providerId, Device device, RuntimeProbe probe) {

    public RuntimeDiagnostic {
        providerId = Objects.requireNonNull(providerId, "providerId");
        device = Objects.requireNonNull(device, "device");
        probe = Objects.requireNonNull(probe, "probe");
    }
}

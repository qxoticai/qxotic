package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public record AvailableDevice(
        Device device, Map<String, String> properties, Set<String> capabilities) {
    public AvailableDevice {
        Objects.requireNonNull(device, "device");
        properties = Map.copyOf(Objects.requireNonNull(properties, "properties"));
        capabilities = Set.copyOf(Objects.requireNonNull(capabilities, "capabilities"));
    }
}

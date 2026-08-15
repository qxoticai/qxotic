package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.List;
import java.util.Set;

public interface RuntimeRegistry {
    DeviceRuntime runtimeFor(Device device);

    boolean hasRuntimeFor(Device device);

    Set<Device> devices();

    List<RuntimeDiagnostic> diagnostics();
}

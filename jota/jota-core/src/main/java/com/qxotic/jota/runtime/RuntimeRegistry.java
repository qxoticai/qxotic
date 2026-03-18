package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.List;
import java.util.Set;

public interface RuntimeRegistry {
    DeviceRuntime runtimeFor(Device device);

    DeviceRuntime runtimeFor(String nameOrAlias);

    boolean hasRuntime(Device device);

    boolean hasRuntime(String nameOrAlias);

    Device resolve(String nameOrAlias);

    Set<Device> devices();

    List<RuntimeDiagnostic> diagnostics();
}

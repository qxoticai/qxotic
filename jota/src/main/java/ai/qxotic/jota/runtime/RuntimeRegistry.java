package ai.qxotic.jota.runtime;

import ai.qxotic.jota.Device;
import java.util.List;
import java.util.Set;

public interface RuntimeRegistry {
    void register(DeviceRuntime deviceRuntime);

    DeviceRuntime runtimeFor(Device device);

    boolean hasRuntime(Device device);

    DeviceRuntime nativeRuntime();

    Set<Device> devices();

    List<RuntimeDiagnostic> diagnostics();

    List<RuntimeDiagnostic> diagnosticsFor(Device device);
}

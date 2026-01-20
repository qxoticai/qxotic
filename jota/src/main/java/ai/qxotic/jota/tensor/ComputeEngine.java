package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;

public interface ComputeEngine {

    ComputeBackend backendFor(Device device);

    KernelCache cache();
}

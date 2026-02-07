package ai.qxotic.jota.runtime.javaaot;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.KernelService;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.lang.foreign.MemorySegment;
import java.util.Optional;

public final class JavaAotDeviceRuntime implements DeviceRuntime {

    private final JavaAotMemoryDomain memoryDomain = new JavaAotMemoryDomain();
    private final ComputeEngine computeEngine = new JavaAotComputeEngine(memoryDomain);

    @Override
    public Device device() {
        return Device.JAVA_AOT;
    }

    @Override
    public MemoryDomain<MemorySegment> memoryDomain() {
        return memoryDomain;
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernelService() {
        return Optional.empty();
    }
}

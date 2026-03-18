package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import java.util.Optional;

/** Mojo backend runtime that uses HIP execution while preserving a distinct backend id. */
public final class MojoDeviceRuntime implements DeviceRuntime {

    private final MojoExecutionEngine<?> executionEngine;
    private final ComputeEngine computeEngine;

    public MojoDeviceRuntime() {
        this.executionEngine = new HipMojoExecutionEngine();
        this.computeEngine = new MojoComputeEngine(executionEngine);
    }

    @Override
    public Device device() {
        return new Device(DeviceType.MOJO, 0);
    }

    @Override
    public MemoryDomain<?> memoryDomain() {
        return executionEngine.memoryDomain();
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernelService() {
        return Optional.of(executionEngine.kernelService());
    }
}

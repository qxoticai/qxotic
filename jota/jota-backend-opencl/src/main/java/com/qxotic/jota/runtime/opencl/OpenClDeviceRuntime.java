package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceCapabilities;
import com.qxotic.jota.runtime.DeviceProperties;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Optional;

public final class OpenClDeviceRuntime implements DeviceRuntime {

    private final OpenClMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public OpenClDeviceRuntime() {
        this(OpenClMemoryDomain.instance());
    }

    public OpenClDeviceRuntime(OpenClMemoryDomain memoryDomain) {
        var probe = new OpenClRuntimeProvider().probe();
        if (!probe.isAvailable()) {
            String hint = probe.hint() == null ? "" : " Hint: " + probe.hint();
            throw new IllegalStateException(
                    "OpenCL backend unavailable ["
                            + probe.status()
                            + "]: "
                            + probe.message()
                            + hint);
        }
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new OpenClComputeEngine(memoryDomain.device());
        KernelBackend backend = new OpenClKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.OPENCL);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<?> memoryDomain() {
        return memoryDomain;
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernelService() {
        return Optional.of(kernelService);
    }

    @Override
    public DeviceProperties properties() {
        var props = new LinkedHashMap<String, Object>();
        props.put(DeviceProperties.DEVICE_NAME, OpenClRuntime.deviceName());
        props.put(DeviceProperties.VENDOR, OpenClRuntime.deviceVendor());
        props.put(DeviceProperties.GLOBAL_MEMORY_BYTES, OpenClRuntime.deviceTotalMem());
        props.put(DeviceProperties.SHARED_MEMORY_BYTES, OpenClRuntime.deviceSharedMemPerBlock());
        props.put(DeviceProperties.MAX_ALLOCATION_BYTES, OpenClRuntime.deviceMaxMemAllocSize());
        props.put(DeviceProperties.COMPUTE_UNITS, (long) OpenClRuntime.deviceComputeUnits());
        props.put(DeviceProperties.CLOCK_MHZ, (long) OpenClRuntime.deviceClockRateMHz());
        props.put(DeviceProperties.MAX_THREADS_PER_BLOCK, OpenClRuntime.deviceMaxThreadsPerBlock());
        return new DeviceProperties(props);
    }

    @Override
    public DeviceCapabilities capabilities() {
        var caps = new LinkedHashSet<String>();
        caps.add(DeviceCapabilities.FP32);
        caps.add(DeviceCapabilities.KERNEL_COMPILATION);
        return new DeviceCapabilities(caps);
    }
}

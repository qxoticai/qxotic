package com.qxotic.jota.runtime.metal;

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

public final class MetalDeviceRuntime implements DeviceRuntime {

    private final MetalMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public MetalDeviceRuntime() {
        this(MetalMemoryDomain.instance());
    }

    public MetalDeviceRuntime(MetalMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new MetalComputeEngine(memoryDomain.device());
        KernelBackend backend = new MetalKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.METAL);
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
        props.put(DeviceProperties.DEVICE_NAME, MetalRuntime.deviceName());
        props.put(DeviceProperties.VENDOR, "Apple");
        props.put(DeviceProperties.ARCHITECTURE, System.getProperty("os.arch"));
        props.put(DeviceProperties.GLOBAL_MEMORY_BYTES, MetalRuntime.deviceTotalMem());
        props.put(DeviceProperties.SHARED_MEMORY_BYTES, MetalRuntime.deviceSharedMemPerBlock());
        props.put(DeviceProperties.MAX_THREADS_PER_BLOCK, MetalRuntime.deviceMaxThreadsPerBlock());
        return new DeviceProperties(props);
    }

    @Override
    public DeviceCapabilities capabilities() {
        var caps = new LinkedHashSet<String>();
        caps.add(DeviceCapabilities.FP16);
        caps.add(DeviceCapabilities.FP32);
        caps.add(DeviceCapabilities.KERNEL_COMPILATION);
        caps.add(DeviceCapabilities.UNIFIED_MEMORY);
        caps.add(DeviceCapabilities.CONCURRENT_KERNELS);
        return new DeviceCapabilities(caps);
    }
}

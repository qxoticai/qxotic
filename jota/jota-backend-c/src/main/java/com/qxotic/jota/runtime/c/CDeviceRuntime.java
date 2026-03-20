package com.qxotic.jota.runtime.c;

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
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Optional;

public final class CDeviceRuntime implements DeviceRuntime {

    private final CMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public CDeviceRuntime() {
        this(DeviceType.C.deviceIndex(0));
    }

    public CDeviceRuntime(Device device) {
        this(new CMemoryDomain(device));
    }

    public CDeviceRuntime(CMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new CComputeEngine(memoryDomain);
        KernelBackend backend = new CKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.C);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
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
        return Optional.of(kernelService);
    }

    @Override
    public boolean supportsNativeRuntimeAlias() {
        return true;
    }

    @Override
    public DeviceProperties properties() {
        var props = new LinkedHashMap<String, Object>();
        Runtime rt = Runtime.getRuntime();
        props.put(DeviceProperties.DEVICE_NAME, "C Host");
        props.put(DeviceProperties.VENDOR, System.getProperty("os.name"));
        props.put(DeviceProperties.ARCHITECTURE, System.getProperty("os.arch"));
        props.put(DeviceProperties.GLOBAL_MEMORY_BYTES, rt.maxMemory());
        props.put(DeviceProperties.COMPUTE_UNITS, (long) rt.availableProcessors());
        return new DeviceProperties(props);
    }

    @Override
    public DeviceCapabilities capabilities() {
        var caps = new LinkedHashSet<String>();
        caps.add(DeviceCapabilities.FP32);
        caps.add(DeviceCapabilities.FP64);
        caps.add(DeviceCapabilities.KERNEL_COMPILATION);
        caps.add(DeviceCapabilities.NATIVE_RUNTIME);
        caps.add(DeviceCapabilities.UNIFIED_MEMORY);
        return new DeviceCapabilities(caps);
    }
}

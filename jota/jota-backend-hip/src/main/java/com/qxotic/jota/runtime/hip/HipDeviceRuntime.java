package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

public final class HipDeviceRuntime implements DeviceRuntime {

    private final HipMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public HipDeviceRuntime() {
        this(DeviceType.HIP.deviceIndex(HipRuntime.currentDevice()));
    }

    public HipDeviceRuntime(Device device) {
        this(new HipMemoryDomain(device));
    }

    public HipDeviceRuntime(HipMemoryDomain memoryDomain) {
        var probe = new HipRuntimeProvider().probe();
        if (!probe.isAvailable()) {
            String hint = probe.hint() == null ? "" : " Hint: " + probe.hint();
            throw new IllegalStateException("HIP backend unavailable: " + probe.message() + hint);
        }
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new HipComputeEngine(memoryDomain.device());
        KernelBackend backend = new HipKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.HIP);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<HipDevicePtr> memoryDomain() {
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
    public Map<String, String> properties() {
        int idx = Math.toIntExact(device().index());
        return Map.of(
                "device.name",
                HipRuntime.deviceName(idx),
                "device.vendor",
                "AMD",
                "device.architecture",
                HipRuntime.deviceArchName(idx),
                "device.kind",
                "gpu");
    }

    @Override
    public Set<String> capabilities() {
        return Set.of(
                "gpu",
                "fp16",
                "fp32",
                "fp64",
                "int8",
                "kernel.compilation",
                "atomic.32",
                "atomic.64");
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + "}";
    }
}

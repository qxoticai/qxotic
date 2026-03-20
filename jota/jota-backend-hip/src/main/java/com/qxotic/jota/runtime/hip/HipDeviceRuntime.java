package com.qxotic.jota.runtime.hip;

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
    public DeviceProperties properties() {
        int idx = Math.toIntExact(device().index());
        var props = new LinkedHashMap<String, Object>();
        props.put(DeviceProperties.DEVICE_NAME, HipRuntime.deviceName(idx));
        props.put(DeviceProperties.VENDOR, "AMD");
        props.put(DeviceProperties.ARCHITECTURE, HipRuntime.deviceArchName(idx));
        props.put(DeviceProperties.GLOBAL_MEMORY_BYTES, HipRuntime.deviceTotalMem(idx));
        props.put(DeviceProperties.SHARED_MEMORY_BYTES, HipRuntime.deviceSharedMemPerBlock(idx));
        props.put(DeviceProperties.COMPUTE_UNITS, (long) HipRuntime.deviceComputeUnits(idx));
        props.put(DeviceProperties.CLOCK_MHZ, (long) (HipRuntime.deviceClockRateKHz(idx) / 1000));
        props.put(DeviceProperties.WARP_SIZE, (long) HipRuntime.deviceWarpSize(idx));
        props.put(
                DeviceProperties.MAX_THREADS_PER_BLOCK,
                (long) HipRuntime.deviceMaxThreadsPerBlock(idx));
        int[] blockDim = HipRuntime.deviceMaxBlockDim(idx);
        props.put(DeviceProperties.MAX_BLOCK_DIM_X, (long) blockDim[0]);
        props.put(DeviceProperties.MAX_BLOCK_DIM_Y, (long) blockDim[1]);
        props.put(DeviceProperties.MAX_BLOCK_DIM_Z, (long) blockDim[2]);
        int[] gridDim = HipRuntime.deviceMaxGridDim(idx);
        props.put(DeviceProperties.MAX_GRID_DIM_X, (long) gridDim[0]);
        props.put(DeviceProperties.MAX_GRID_DIM_Y, (long) gridDim[1]);
        props.put(DeviceProperties.MAX_GRID_DIM_Z, (long) gridDim[2]);
        props.put(
                DeviceProperties.MAX_REGISTERS_PER_BLOCK,
                (long) HipRuntime.deviceRegsPerBlock(idx));
        props.put(DeviceProperties.L2_CACHE_BYTES, (long) HipRuntime.deviceL2CacheSize(idx));
        props.put(
                DeviceProperties.MEMORY_BUS_WIDTH_BITS,
                (long) HipRuntime.deviceMemoryBusWidthBits(idx));
        props.put(
                DeviceProperties.MEMORY_CLOCK_MHZ,
                (long) (HipRuntime.deviceMemoryClockRateKHz(idx) / 1000));
        return new DeviceProperties(props);
    }

    @Override
    public DeviceCapabilities capabilities() {
        int idx = Math.toIntExact(device().index());
        var caps = new LinkedHashSet<String>();
        caps.add(DeviceCapabilities.FP16);
        caps.add(DeviceCapabilities.FP32);
        caps.add(DeviceCapabilities.FP64);
        caps.add(DeviceCapabilities.INT8);
        caps.add(DeviceCapabilities.KERNEL_COMPILATION);
        caps.add(DeviceCapabilities.ATOMIC_32);
        caps.add(DeviceCapabilities.ATOMIC_64);
        if (HipRuntime.deviceConcurrentKernels(idx)) {
            caps.add(DeviceCapabilities.CONCURRENT_KERNELS);
        }
        if (HipRuntime.deviceEccEnabled(idx)) {
            caps.add(DeviceCapabilities.ECC_MEMORY);
        }
        return new DeviceCapabilities(caps);
    }
}

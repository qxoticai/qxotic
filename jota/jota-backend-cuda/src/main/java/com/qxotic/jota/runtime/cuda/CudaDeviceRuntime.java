package com.qxotic.jota.runtime.cuda;

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

public final class CudaDeviceRuntime implements DeviceRuntime {

    private final CudaMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public CudaDeviceRuntime() {
        this(CudaMemoryDomain.instance());
    }

    public CudaDeviceRuntime(CudaMemoryDomain memoryDomain) {
        var probe = new CudaRuntimeProvider().probe();
        if (!probe.isAvailable()) {
            String hint = probe.hint() == null ? "" : " Hint: " + probe.hint();
            throw new IllegalStateException("CUDA backend unavailable: " + probe.message() + hint);
        }
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new CudaComputeEngine(memoryDomain.device());
        KernelBackend backend = new CudaKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.CUDA);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<CudaDevicePtr> memoryDomain() {
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
        int idx = device().index();
        var props = new LinkedHashMap<String, Object>();
        props.put(DeviceProperties.DEVICE_NAME, CudaRuntime.deviceName(idx));
        props.put(DeviceProperties.VENDOR, "NVIDIA");
        props.put(DeviceProperties.ARCHITECTURE, CudaRuntime.deviceArchName(idx));
        props.put(DeviceProperties.GLOBAL_MEMORY_BYTES, CudaRuntime.deviceTotalMem(idx));
        props.put(DeviceProperties.SHARED_MEMORY_BYTES, CudaRuntime.deviceSharedMemPerBlock(idx));
        props.put(DeviceProperties.COMPUTE_UNITS, (long) CudaRuntime.deviceComputeUnits(idx));
        props.put(DeviceProperties.CLOCK_MHZ, (long) (CudaRuntime.deviceClockRateKHz(idx) / 1000));
        props.put(DeviceProperties.WARP_SIZE, (long) CudaRuntime.deviceWarpSize(idx));
        props.put(
                DeviceProperties.MAX_THREADS_PER_BLOCK,
                (long) CudaRuntime.deviceMaxThreadsPerBlock(idx));
        int[] blockDim = CudaRuntime.deviceMaxBlockDim(idx);
        props.put(DeviceProperties.MAX_BLOCK_DIM_X, (long) blockDim[0]);
        props.put(DeviceProperties.MAX_BLOCK_DIM_Y, (long) blockDim[1]);
        props.put(DeviceProperties.MAX_BLOCK_DIM_Z, (long) blockDim[2]);
        int[] gridDim = CudaRuntime.deviceMaxGridDim(idx);
        props.put(DeviceProperties.MAX_GRID_DIM_X, (long) gridDim[0]);
        props.put(DeviceProperties.MAX_GRID_DIM_Y, (long) gridDim[1]);
        props.put(DeviceProperties.MAX_GRID_DIM_Z, (long) gridDim[2]);
        props.put(
                DeviceProperties.MAX_REGISTERS_PER_BLOCK,
                (long) CudaRuntime.deviceRegsPerBlock(idx));
        props.put(DeviceProperties.L2_CACHE_BYTES, (long) CudaRuntime.deviceL2CacheSize(idx));
        props.put(
                DeviceProperties.MEMORY_BUS_WIDTH_BITS,
                (long) CudaRuntime.deviceMemoryBusWidthBits(idx));
        props.put(
                DeviceProperties.MEMORY_CLOCK_MHZ,
                (long) (CudaRuntime.deviceMemoryClockRateKHz(idx) / 1000));
        return new DeviceProperties(props);
    }

    @Override
    public DeviceCapabilities capabilities() {
        int idx = device().index();
        var caps = new LinkedHashSet<String>();
        caps.add(DeviceCapabilities.FP32);
        String arch = CudaRuntime.deviceArchName(idx);
        if (arch != null && arch.compareTo("sm_53") >= 0) {
            caps.add(DeviceCapabilities.FP16);
        }
        if (arch != null && arch.compareTo("sm_80") >= 0) {
            caps.add(DeviceCapabilities.BF16);
            caps.add(DeviceCapabilities.TF32);
        }
        caps.add(DeviceCapabilities.FP64);
        caps.add(DeviceCapabilities.INT8);
        caps.add(DeviceCapabilities.KERNEL_COMPILATION);
        caps.add(DeviceCapabilities.ATOMIC_32);
        caps.add(DeviceCapabilities.ATOMIC_64);
        if (CudaRuntime.deviceConcurrentKernels(idx)) {
            caps.add(DeviceCapabilities.CONCURRENT_KERNELS);
        }
        if (CudaRuntime.deviceEccEnabled(idx)) {
            caps.add(DeviceCapabilities.ECC_MEMORY);
        }
        if (CudaRuntime.deviceUnifiedAddressing(idx)) {
            caps.add(DeviceCapabilities.UNIFIED_MEMORY);
        }
        return new DeviceCapabilities(caps);
    }
}

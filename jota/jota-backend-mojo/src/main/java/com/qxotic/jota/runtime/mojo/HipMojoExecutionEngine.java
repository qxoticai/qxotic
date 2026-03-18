package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.hip.HipDevicePtr;
import com.qxotic.jota.runtime.hip.HipDeviceRuntime;
import java.nio.file.Path;

/** HIP-backed execution engine for Mojo frontend generated kernels. */
final class HipMojoExecutionEngine implements MojoExecutionEngine<HipDevicePtr> {

    private final MojoMemoryDomain<HipDevicePtr> memoryDomain;
    private final KernelService kernelService;

    @SuppressWarnings("unchecked")
    HipMojoExecutionEngine() {
        HipDeviceRuntime hipRuntime = new HipDeviceRuntime();
        this.memoryDomain =
                new MojoMemoryDomain((MemoryDomain<HipDevicePtr>) hipRuntime.memoryDomain());
        KernelBackend backend =
                new MojoKernelBackend(
                        hipRuntime
                                .kernelService()
                                .orElseThrow(
                                        () ->
                                                new IllegalStateException(
                                                        "HIP kernel service unavailable"))
                                .backend());
        Path programRoot = KernelCachePaths.programRoot(DeviceType.MOJO);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public MojoMemoryDomain<HipDevicePtr> memoryDomain() {
        return memoryDomain;
    }

    @Override
    public long addressOf(HipDevicePtr pointer) {
        return pointer.address();
    }

    @Override
    public KernelService kernelService() {
        return kernelService;
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey key) {
        return kernelService.backend().getOrCompile(program, key);
    }
}

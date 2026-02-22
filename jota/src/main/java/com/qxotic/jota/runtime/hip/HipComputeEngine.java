package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRStandardPipeline;
import com.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.panama.LIRKernelArgsBuilder;
import com.qxotic.jota.tensor.*;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class HipComputeEngine implements ComputeEngine {

    private final Device device;
    private final HipKernelProgramGenerator generator = new HipKernelProgramGenerator();
    private final HipKernelBackend backend = new HipKernelBackend();
    private final LIRKernelArgsBuilder lirArgsBuilder = new LIRKernelArgsBuilder();
    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));

    HipComputeEngine(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        if (!HipRuntime.isAvailable()) {
            throw new IllegalStateException("HIP runtime not available");
        }
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph));
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }

        HipMemoryDomain hipContext = HipMemoryDomain.instance();
        List<MemoryView<?>> outputs = allocateOutputs(lirGraph, hipContext);

        Memory<HipDevicePtr> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch =
                    hipContext
                            .memoryAllocator()
                            .allocateMemory(scratchLayout.alignedTotalByteSize(), 64L);
        }

        KernelCacheKey key = backend.cacheKey(lirGraph, scratchLayout);
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);
        KernelArgs args = lirArgsBuilder.build(lirGraph, inputs, outputs);
        long scratchPtr = scratch == null ? 0L : scratch.base().address();
        args.addScalarBits(scratchPtr, DataType.I64);
        LaunchConfig config = chooseLirLaunchConfig(lirGraph);
        ExecutionStream stream = new ExecutionStream(Device.HIP, 0L, true);
        KernelExecutable exec = backend.getOrCompile(program, key);
        exec.launch(config, args, stream);

        if (shouldReturnHostOutput()) {
            return toHost(hipContext, castDevice(outputs.getFirst()));
        }
        return outputs.getFirst();
    }

    private static MemoryView<HipDevicePtr> toDevice(
            HipMemoryDomain hipContext, MemoryView<?> view) {
        if (view.memory().device().equals(Device.HIP)) {
            @SuppressWarnings("unchecked")
            MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
            return hipView;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcContext =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryView<HipDevicePtr> dst =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(view.dataType(), view.shape().size()),
                        view.dataType(),
                        Layout.rowMajor(view.shape()));
        MemoryDomain.copy(srcContext, srcView, hipContext, dst);
        return dst;
    }

    private static MemoryView<HipDevicePtr> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
        return hipView;
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryDomain<MemorySegment> hostContext, MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment
                && view.memory().device().belongsTo(Device.CPU)) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) view;
            return hostView;
        }
        MemoryView<MemorySegment> dst =
                MemoryView.of(
                        hostContext.memoryAllocator().allocateMemory(view.dataType(), view.shape()),
                        view.dataType(),
                        view.layout());
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcContext =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryDomain.copy(srcContext, srcView, hostContext, dst);
        return dst;
    }

    private static boolean shouldReturnHostOutput() {
        Device defaultDevice = Environment.current().defaultDevice();
        return defaultDevice.belongsTo(Device.CPU);
    }

    private static MemoryView<?> toHost(HipMemoryDomain hipContext, MemoryView<HipDevicePtr> view) {
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> hostContext =
                (MemoryDomain<Object>) Environment.current().nativeRuntime().memoryDomain();
        @SuppressWarnings("unchecked")
        MemoryView<Object> hostView =
                (MemoryView<Object>)
                        MemoryView.of(
                                hostContext
                                        .memoryAllocator()
                                        .allocateMemory(view.dataType(), view.shape().size()),
                                view.dataType(),
                                Layout.rowMajor(view.shape()));
        MemoryDomain.copy(hipContext, view, hostContext, hostView);
        return hostView;
    }

    private static List<MemoryView<?>> allocateOutputs(
            LIRGraph graph, HipMemoryDomain memoryDomain) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<HipDevicePtr> memory =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(output.dataType(), output.shape());
            outputs.add(MemoryView.of(memory, 0, output.dataType(), output.layout()));
        }
        return outputs;
    }

    private static LaunchConfig chooseLirLaunchConfig(LIRGraph graph) {
        long workSize = 1L;
        if (!graph.outputs().isEmpty()) {
            workSize = graph.outputs().getFirst().size();
        }
        int threads = 256;
        long blocksLong = (workSize + threads - 1) / threads;
        if (blocksLong < 1) {
            blocksLong = 1;
        }
        int blocks = blocksLong > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) blocksLong;
        return new LaunchConfig(blocks, 1, 1, threads, 1, 1, 0, false);
    }
}

package ai.qxotic.jota.hip;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRStandardPipeline;
import ai.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.panama.LIRKernelArgsBuilder;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import ai.qxotic.jota.tensor.LaunchConfig;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class HipComputeBackend implements ai.qxotic.jota.tensor.ComputeBackend {

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

    HipComputeBackend(Device device) {
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

        HipMemoryContext hipContext = HipMemoryContext.instance();
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
            HipMemoryContext hipContext, MemoryView<?> view) {
        if (view.memory().device().equals(Device.HIP)) {
            @SuppressWarnings("unchecked")
            MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
            return hipView;
        }
        @SuppressWarnings("unchecked")
        MemoryContext<Object> srcContext =
                (MemoryContext<Object>)
                        Environment.current().backend(view.memory().device()).memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryView<HipDevicePtr> dst =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(view.dataType(), view.shape().size()),
                        view.dataType(),
                        Layout.rowMajor(view.shape()));
        MemoryContext.copy(srcContext, srcView, hipContext, dst);
        return dst;
    }

    private static MemoryView<HipDevicePtr> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
        return hipView;
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryContext<MemorySegment> hostContext, MemoryView<?> view) {
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
        MemoryContext<Object> srcContext =
                (MemoryContext<Object>)
                        Environment.current().backend(view.memory().device()).memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryContext.copy(srcContext, srcView, hostContext, dst);
        return dst;
    }

    private static boolean shouldReturnHostOutput() {
        Device defaultDevice = Environment.current().defaultDevice();
        return defaultDevice.belongsTo(Device.CPU);
    }

    private static MemoryView<?> toHost(
            HipMemoryContext hipContext, MemoryView<HipDevicePtr> view) {
        @SuppressWarnings("unchecked")
        MemoryContext<Object> hostContext =
                (MemoryContext<Object>) Environment.current().nativeBackend().memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> hostView =
                (MemoryView<Object>)
                        MemoryView.of(
                                hostContext
                                        .memoryAllocator()
                                        .allocateMemory(view.dataType(), view.shape().size()),
                                view.dataType(),
                                Layout.rowMajor(view.shape()));
        MemoryContext.copy(hipContext, view, hostContext, hostView);
        return hostView;
    }

    private static List<MemoryView<?>> allocateOutputs(
            LIRGraph graph, HipMemoryContext context) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<HipDevicePtr> memory =
                    context.memoryAllocator().allocateMemory(output.dataType(), output.shape());
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

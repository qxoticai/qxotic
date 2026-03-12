package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.LIRStandardPipeline;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TIRTextRenderer;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.*;
import com.qxotic.jota.runtime.clike.CLikeKernelGenerator;
import com.qxotic.jota.runtime.clike.LIRKernelArgsBuilder;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class CudaComputeEngine implements ComputeEngine {

    private final Device device;
    private final CLikeKernelGenerator generator = new CLikeKernelGenerator(new CudaDialect());
    private final CudaKernelBackend backend = new CudaKernelBackend();
    private final LIRKernelArgsBuilder lirArgsBuilder = new LIRKernelArgsBuilder();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final boolean logKernelIr = Boolean.getBoolean("jota.kernel.ir.log");
    private static final boolean LOG_TIMING = Boolean.getBoolean("jota.cuda.timing.log");

    CudaComputeEngine(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        if (!CudaRuntime.isAvailable()) {
            throw new IllegalStateException("CUDA runtime not available");
        }
        long t0 = System.nanoTime();
        List<MemoryView<?>> runtimeInputViews = materializeRuntimeInputViews(graph, inputs);
        long tMatInputs = System.nanoTime();
        List<Layout> runtimeInputLayouts = runtimeInputLayouts(graph, runtimeInputViews);
        long tInputLayouts = System.nanoTime();
        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, runtimeInputLayouts));
        long tLowerAndPipeline = System.nanoTime();
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        long tScratchAnalyze = System.nanoTime();
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }
        long tScratchVerify = System.nanoTime();

        CudaMemoryDomain hipContext = CudaMemoryDomain.instance();
        List<MemoryView<?>> outputs = allocateOutputs(lirGraph, hipContext);
        long tAllocateOutputs = System.nanoTime();

        Memory<CudaDevicePtr> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch =
                    hipContext
                            .memoryAllocator()
                            .allocateMemory(scratchLayout.alignedTotalByteSize(), 64L);
        }
        long tAllocateScratch = System.nanoTime();

        KernelCacheKey key = backend.cacheKey(lirGraph, scratchLayout);
        logKernelIr(key, graph, lirGraph);
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);
        long tGenerateProgram = System.nanoTime();
        List<Tensor> resolvedInputs =
                resolveInputs(lirGraph, inputs, runtimeInputViews, hipContext);
        long tResolveInputs = System.nanoTime();
        long scratchPtr = scratch == null ? 0L : scratch.base().address();
        KernelArgs args =
                lirArgsBuilder.buildGroupedWithWorkspaceScalar(
                        lirGraph, resolvedInputs, outputs, scratchPtr, DataType.I64);
        long tBuildArgs = System.nanoTime();
        LaunchConfig config = chooseLirLaunchConfig(lirGraph);
        ExecutionStream stream = new ExecutionStream(Device.CUDA, 0L, true);
        KernelExecutable exec = backend.getOrCompile(program, key);
        long tGetOrCompile = System.nanoTime();
        exec.launch(config, args, stream);
        long tLaunch = System.nanoTime();

        if (shouldReturnHostOutput()) {
            MemoryView<?> host = toHost(hipContext, castDevice(outputs.getFirst()));
            long tToHost = System.nanoTime();
            logTiming(
                    key,
                    t0,
                    tMatInputs,
                    tInputLayouts,
                    tLowerAndPipeline,
                    tScratchAnalyze,
                    tScratchVerify,
                    tAllocateOutputs,
                    tAllocateScratch,
                    tGenerateProgram,
                    tResolveInputs,
                    tBuildArgs,
                    tGetOrCompile,
                    tLaunch,
                    tToHost);
            return host;
        }
        logTiming(
                key,
                t0,
                tMatInputs,
                tInputLayouts,
                tLowerAndPipeline,
                tScratchAnalyze,
                tScratchVerify,
                tAllocateOutputs,
                tAllocateScratch,
                tGenerateProgram,
                tResolveInputs,
                tBuildArgs,
                tGetOrCompile,
                tLaunch,
                tLaunch);
        return outputs.getFirst();
    }

    @Override
    public boolean supportsParallelPrecompile() {
        return true;
    }

    @Override
    public void precompile(TIRGraph graph) {
        List<Layout> inputLayouts = graphInputLayouts(graph);
        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, inputLayouts));
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }
        KernelCacheKey key = backend.cacheKey(lirGraph, scratchLayout);
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);
        backend.getOrCompile(program, key);
    }

    private static void logTiming(
            KernelCacheKey key,
            long t0,
            long tMatInputs,
            long tInputLayouts,
            long tLowerAndPipeline,
            long tScratchAnalyze,
            long tScratchVerify,
            long tAllocateOutputs,
            long tAllocateScratch,
            long tGenerateProgram,
            long tResolveInputs,
            long tBuildArgs,
            long tGetOrCompile,
            long tLaunch,
            long tEnd) {
        if (!LOG_TIMING) {
            return;
        }
        System.out.println(
                "[jota-cuda-timing] key="
                        + key.value()
                        + " materializeInputsMs="
                        + ms(tMatInputs - t0)
                        + " inputLayoutsMs="
                        + ms(tInputLayouts - tMatInputs)
                        + " lowerPipelineMs="
                        + ms(tLowerAndPipeline - tInputLayouts)
                        + " scratchAnalyzeMs="
                        + ms(tScratchAnalyze - tLowerAndPipeline)
                        + " scratchVerifyMs="
                        + ms(tScratchVerify - tScratchAnalyze)
                        + " allocOutputsMs="
                        + ms(tAllocateOutputs - tScratchVerify)
                        + " allocScratchMs="
                        + ms(tAllocateScratch - tAllocateOutputs)
                        + " programGenerateMs="
                        + ms(tGenerateProgram - tAllocateScratch)
                        + " resolveInputsMs="
                        + ms(tResolveInputs - tGenerateProgram)
                        + " buildArgsMs="
                        + ms(tBuildArgs - tResolveInputs)
                        + " getOrCompileMs="
                        + ms(tGetOrCompile - tBuildArgs)
                        + " launchMs="
                        + ms(tLaunch - tGetOrCompile)
                        + " returnToHostMs="
                        + ms(tEnd - tLaunch)
                        + " totalMs="
                        + ms(tEnd - t0));
    }

    private static String ms(long nanos) {
        return String.format(Locale.ROOT, "%.3f", nanos / 1_000_000.0);
    }

    private static List<Layout> graphInputLayouts(TIRGraph graph) {
        List<Layout> layouts = new ArrayList<>(graph.inputs().size());
        for (TIRNode input : graph.inputs()) {
            if (input instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                layouts.add(null);
                continue;
            }
            if (input instanceof TensorInput tensorInput) {
                layouts.add(tensorInput.layout());
                continue;
            }
            throw new IllegalArgumentException(
                    "Unsupported TIR input node for precompile: "
                            + input.getClass().getSimpleName());
        }
        return layouts;
    }

    private static List<MemoryView<?>> materializeRuntimeInputViews(
            TIRGraph graph, List<Tensor> inputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        List<MemoryView<?>> materialized = new ArrayList<>(inputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            if (graph.inputs().get(i) instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                materialized.add(null);
                continue;
            }
            Tensor tensor = inputs.get(i);
            materialized.add(tensor.materialize());
        }
        return materialized;
    }

    private static List<Layout> runtimeInputLayouts(
            TIRGraph graph, List<MemoryView<?>> runtimeViews) {
        List<Layout> layouts = new ArrayList<>(graph.inputs().size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            if (graph.inputs().get(i) instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                layouts.add(null);
                continue;
            }
            MemoryView<?> view = runtimeViews.get(i);
            layouts.add(view.layout());
        }
        return layouts;
    }

    private static List<Tensor> resolveInputs(
            LIRGraph graph,
            List<Tensor> originalInputs,
            List<MemoryView<?>> runtimeInputViews,
            CudaMemoryDomain hipContext) {
        List<Tensor> resolved = new ArrayList<>(originalInputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof ScalarInput) {
                resolved.add(originalInputs.get(i));
                continue;
            }
            MemoryView<?> runtimeView = runtimeInputViews.get(i);
            resolved.add(Tensor.of(toDevice(hipContext, runtimeView)));
        }
        return resolved;
    }

    private void logKernelIr(KernelCacheKey key, TIRGraph tirGraph, LIRGraph lirGraph) {
        if (!logKernelIr) {
            return;
        }
        System.out.println("[jota-ir] BEGIN CUDA kernel key=" + key.value());
        System.out.println("[jota-ir] TIR:\n" + new TIRTextRenderer().render(tirGraph));
        System.out.println("[jota-ir] LIR:\n" + new LIRTextRenderer().render(lirGraph));
        System.out.println("[jota-ir] END CUDA kernel key=" + key.value());
    }

    private static MemoryView<CudaDevicePtr> toDevice(
            CudaMemoryDomain hipContext, MemoryView<?> view) {
        if (view.memory().device().equals(Device.CUDA)) {
            @SuppressWarnings("unchecked")
            MemoryView<CudaDevicePtr> hipView = (MemoryView<CudaDevicePtr>) view;
            return hipView;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcContext =
                Environment.current().memoryDomainFor(view.memory().device());
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        BufferSpec spec = computeBufferSpec(view.layout(), view.dataType());
        MemoryView<CudaDevicePtr> dst =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(spec.byteSize()),
                        spec.byteOffset(),
                        view.dataType(),
                        view.layout());
        MemoryDomain.copy(srcContext, srcView, hipContext, dst);
        return dst;
    }

    private static MemoryView<CudaDevicePtr> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<CudaDevicePtr> hipView = (MemoryView<CudaDevicePtr>) view;
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
                Environment.current().memoryDomainFor(view.memory().device());
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryDomain.copy(srcContext, srcView, hostContext, dst);
        return dst;
    }

    private static boolean shouldReturnHostOutput() {
        Device defaultDevice = Environment.current().defaultDevice();
        return defaultDevice.belongsTo(Device.CPU);
    }

    private static MemoryView<?> toHost(
            CudaMemoryDomain hipContext, MemoryView<CudaDevicePtr> view) {
        MemoryDomain<MemorySegment> hostContext = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView =
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
            LIRGraph graph, CudaMemoryDomain memoryDomain) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<CudaDevicePtr> memory =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(output.dataType(), output.shape());
            outputs.add(MemoryView.of(memory, 0, output.dataType(), output.layout()));
        }
        return outputs;
    }

    private static BufferSpec computeBufferSpec(Layout layout, DataType dataType) {
        long[] shape = layout.shape().toArray();
        long[] strideBytes = layout.stride().scale(dataType.byteSize()).toArray();
        long minOffset = 0;
        long maxOffset = 0;
        for (int i = 0; i < shape.length; i++) {
            long dim = shape[i];
            if (dim <= 1) {
                continue;
            }
            long span = (dim - 1) * strideBytes[i];
            if (strideBytes[i] >= 0) {
                maxOffset += span;
            } else {
                minOffset += span;
            }
        }
        long byteOffset = -minOffset;
        long byteSize = maxOffset - minOffset + dataType.byteSize();
        return new BufferSpec(byteOffset, byteSize);
    }

    private record BufferSpec(long byteOffset, long byteSize) {}

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

package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.LIRStandardPipeline;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRInterpreter;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.runtime.clike.CLikeKernelGenerator;
import com.qxotic.jota.runtime.panama.LIRKernelArgsBuilder;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class OpenClComputeEngine implements ComputeEngine {

    private final Device device;
    private final CLikeKernelGenerator generator = new CLikeKernelGenerator(new OpenClDialect());
    private final OpenClKernelBackend backend = new OpenClKernelBackend();
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final boolean interpreterFallback =
            Boolean.parseBoolean(System.getProperty("jota.opencl.interpreter.fallback", "true"));

    OpenClComputeEngine(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        OpenClRuntime.requireAvailable();
        if (interpreterFallback) {
            try {
                return executeCompiled(graph, inputs);
            } catch (UnsupportedOperationException e) {
                return executeWithInterpreterFallback(graph, inputs);
            }
        }
        return executeCompiled(graph, inputs);
    }

    private MemoryView<?> executeCompiled(TIRGraph graph, List<Tensor> inputs) {
        List<MemoryView<?>> runtimeInputViews = materializeRuntimeInputViews(graph, inputs);
        List<Layout> runtimeInputLayouts = runtimeInputLayouts(graph, runtimeInputViews);
        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, runtimeInputLayouts));
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }

        OpenClMemoryDomain openclDomain = OpenClMemoryDomain.instance();
        List<MemoryView<?>> outputs = allocateOutputs(lirGraph, openclDomain);
        KernelCacheKey key = backend.cacheKey(lirGraph, scratchLayout);
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);
        List<Tensor> resolvedInputs =
                resolveInputs(lirGraph, inputs, runtimeInputViews, openclDomain);
        KernelArgs args = argsBuilder.build(lirGraph, resolvedInputs, outputs);
        args.addBuffer(allocateScratchBuffer(openclDomain, scratchLayout));
        LaunchConfig config = chooseLirLaunchConfig(lirGraph);
        ExecutionStream stream = new ExecutionStream(Device.OPENCL, 0L, true);
        KernelExecutable exec = backend.getOrCompile(program, key);
        exec.launch(config, args, stream);

        if (shouldReturnHostOutput()) {
            return toHost(openclDomain, castDevice(outputs.getFirst()));
        }
        return outputs.getFirst();
    }

    private MemoryView<?> executeWithInterpreterFallback(TIRGraph graph, List<Tensor> inputs) {
        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        List<MemoryView<?>> hostInputs = new ArrayList<>(inputs.size());
        for (var input : inputs) {
            MemoryView<?> view = input.materialize();
            hostInputs.add(toHost(hostDomain, view));
        }
        List<MemoryView<MemorySegment>> hostOutputs =
                TIRInterpreter.execute(graph, hostInputs, hostDomain);
        if (hostOutputs.isEmpty()) {
            throw new IllegalStateException("TIR interpreter returned no outputs");
        }
        if (shouldReturnHostOutput()) {
            return hostOutputs.getFirst();
        }
        return toDevice(OpenClMemoryDomain.instance(), hostOutputs.getFirst());
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

    private static MemoryView<OpenClDevicePtr> allocateScratchBuffer(
            OpenClMemoryDomain domain, ScratchLayout scratchLayout) {
        long bytes = scratchLayout.requiresScratch() ? scratchLayout.alignedTotalByteSize() : 1L;
        Memory<OpenClDevicePtr> memory = domain.memoryAllocator().allocateMemory(bytes, 1);
        return MemoryView.of(
                memory, 0, DataType.I8, Layout.rowMajor(com.qxotic.jota.Shape.of(bytes)));
    }

    private static List<Layout> graphInputLayouts(TIRGraph graph) {
        List<Layout> layouts = new ArrayList<>(graph.inputs().size());
        for (var input : graph.inputs()) {
            if (input instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                layouts.add(null);
                continue;
            }
            if (input instanceof com.qxotic.jota.ir.tir.TensorInput tensorInput) {
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
            materialized.add(inputs.get(i).materialize());
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
            layouts.add(runtimeViews.get(i).layout());
        }
        return layouts;
    }

    private static List<Tensor> resolveInputs(
            LIRGraph graph,
            List<Tensor> originalInputs,
            List<MemoryView<?>> runtimeInputViews,
            OpenClMemoryDomain openclDomain) {
        List<Tensor> resolved = new ArrayList<>(originalInputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof ScalarInput) {
                resolved.add(originalInputs.get(i));
                continue;
            }
            MemoryView<?> runtimeView = runtimeInputViews.get(i);
            resolved.add(Tensor.of(toDevice(openclDomain, runtimeView)));
        }
        return resolved;
    }

    private static MemoryView<OpenClDevicePtr> toDevice(
            OpenClMemoryDomain openclDomain, MemoryView<?> view) {
        if (view.memory().device().equals(Device.OPENCL)) {
            @SuppressWarnings("unchecked")
            MemoryView<OpenClDevicePtr> openclView = (MemoryView<OpenClDevicePtr>) view;
            return openclView;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().memoryDomainFor(view.memory().device());
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        BufferSpec spec = computeBufferSpec(view.layout(), view.dataType());
        MemoryView<OpenClDevicePtr> dst =
                MemoryView.of(
                        openclDomain.memoryAllocator().allocateMemory(spec.byteSize()),
                        spec.byteOffset(),
                        view.dataType(),
                        view.layout());
        MemoryDomain.copy(srcDomain, srcView, openclDomain, dst);
        return dst;
    }

    private static MemoryView<MemorySegment> toHost(
            OpenClMemoryDomain openclDomain, MemoryView<OpenClDevicePtr> view) {
        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        hostDomain
                                .memoryAllocator()
                                .allocateMemory(view.dataType(), view.shape().size()),
                        view.dataType(),
                        Layout.rowMajor(view.shape()));
        MemoryDomain.copy(openclDomain, view, hostDomain, hostView);
        return hostView;
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryDomain<MemorySegment> hostDomain, MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) view;
            return hostView;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().memoryDomainFor(view.memory().device());
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        BufferSpec spec = computeBufferSpec(view.layout(), view.dataType());
        MemoryView<MemorySegment> dst =
                MemoryView.of(
                        hostDomain.memoryAllocator().allocateMemory(spec.byteSize()),
                        spec.byteOffset(),
                        view.dataType(),
                        view.layout());
        MemoryDomain.copy(srcDomain, srcView, hostDomain, dst);
        return dst;
    }

    private static MemoryView<OpenClDevicePtr> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<OpenClDevicePtr> openclView = (MemoryView<OpenClDevicePtr>) view;
        return openclView;
    }

    private static boolean shouldReturnHostOutput() {
        return Environment.current().defaultDevice().belongsTo(Device.CPU);
    }

    private static List<MemoryView<?>> allocateOutputs(
            LIRGraph graph, OpenClMemoryDomain openclDomain) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<OpenClDevicePtr> memory =
                    openclDomain
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

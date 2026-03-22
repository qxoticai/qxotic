package com.qxotic.jota.runtime.c;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
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
import com.qxotic.jota.ir.tir.TIRTextRenderer;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.runtime.clike.CLikeKernelGenerator;
import com.qxotic.jota.runtime.clike.LIRKernelArgsBuilder;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class CComputeEngine implements ComputeEngine {

    private final CMemoryDomain memoryDomain;
    private final CKernelBackend backend = new CKernelBackend();
    private final CLikeKernelGenerator generator = new CLikeKernelGenerator(new CDialect());
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final boolean logKernelIr = Boolean.getBoolean("jota.kernel.ir.log");
    private final Device device;

    public CComputeEngine(CMemoryDomain memoryDomain) {
        this.memoryDomain = memoryDomain;
        this.device = memoryDomain.device();
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        CNative.requireAvailable();
        List<MemoryView<?>> runtimeInputViews = materializeRuntimeInputViews(graph, inputs);
        List<Layout> runtimeInputLayouts = runtimeInputLayouts(graph, runtimeInputViews);

        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, runtimeInputLayouts));
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }

        List<MemoryView<?>> outputViews = allocateOutputs(lirGraph, memoryDomain);
        Memory<MemorySegment> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(scratchLayout.alignedTotalByteSize(), 64L);
        }

        KernelCacheKey key = backend.cacheKey(lirGraph, scratchLayout);
        logKernelIr(key, graph, lirGraph);
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);

        List<Tensor> resolvedInputs = resolveInputs(lirGraph, inputs, runtimeInputViews);
        KernelArgs args = argsBuilder.build(lirGraph, resolvedInputs, outputViews);
        long scratchPtr = scratch == null ? 0L : scratch.base().address();
        args.addMetadata(scratchPtr);

        CKernelExecutable exec = (CKernelExecutable) backend.getOrCompile(program, key);
        exec.launch(
                new LaunchConfig(1, 1, 1, 1, 1, 1, 0, false),
                args,
                new ExecutionStream(device, null, true));

        return outputViews.getFirst();
    }

    private void logKernelIr(KernelCacheKey key, TIRGraph tirGraph, LIRGraph lirGraph) {
        if (!logKernelIr) {
            return;
        }
        System.out.println("[jota-ir] BEGIN C kernel key=" + key.value());
        System.out.println("[jota-ir] TIR:\n" + new TIRTextRenderer().render(tirGraph));
        System.out.println("[jota-ir] LIR:\n" + new LIRTextRenderer().render(lirGraph));
        System.out.println("[jota-ir] END C kernel key=" + key.value());
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

    private List<Tensor> resolveInputs(
            LIRGraph graph, List<Tensor> inputs, List<MemoryView<?>> runtimeInputViews) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        List<Tensor> resolved = new ArrayList<>(inputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            Tensor tensor = inputs.get(i);
            if (input instanceof ScalarInput) {
                resolved.add(tensor);
                continue;
            }

            MemoryView<?> view = runtimeInputViews.get(i);
            @SuppressWarnings("unchecked")
            MemoryDomain<Object> srcContext =
                    (MemoryDomain<Object>)
                            Environment.runtimeFor(view.memory().device()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryView<Object> srcView = (MemoryView<Object>) view;

            if (view.memory().device().belongsTo(DeviceType.C)
                    && view.memory().base() instanceof MemorySegment) {
                resolved.add(Tensor.of(view));
                continue;
            }

            BufferSpec spec = computeBufferSpec(view.layout(), view.dataType());
            MemoryView<MemorySegment> copy =
                    MemoryView.of(
                            memoryDomain.memoryAllocator().allocateMemory(spec.byteSize()),
                            spec.byteOffset(),
                            view.dataType(),
                            view.layout());
            MemoryDomain.copy(srcContext, srcView, memoryDomain, copy);
            resolved.add(Tensor.of(copy));
        }
        return resolved;
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

    private static List<MemoryView<?>> allocateOutputs(LIRGraph graph, CMemoryDomain memoryDomain) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<MemorySegment> memory =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(output.dataType(), output.shape());
            outputs.add(MemoryView.of(memory, 0, output.dataType(), output.layout()));
        }
        return outputs;
    }
}

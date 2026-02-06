package ai.qxotic.jota.c;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRInput;
import ai.qxotic.jota.ir.lir.LIRStandardPipeline;
import ai.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.panama.LIRKernelArgsBuilder;
import ai.qxotic.jota.tensor.*;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class CComputeEngine implements ComputeEngine {

    private final CMemoryDomain memoryDomain;
    private final CKernelBackend backend = new CKernelBackend();
    private final CKernelProgramGenerator generator = new CKernelProgramGenerator();
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));

    public CComputeEngine(CMemoryDomain memoryDomain) {
        this.memoryDomain = memoryDomain;
    }

    @Override
    public Device device() {
        return Device.C;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        CNative.requireAvailable();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph));
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
        KernelProgram program = generator.generate(lirGraph, scratchLayout, key);

        List<Tensor> resolvedInputs = resolveInputs(lirGraph, inputs);
        KernelArgs args = argsBuilder.build(lirGraph, resolvedInputs, outputViews);
        long scratchPtr = scratch == null ? 0L : scratch.base().address();
        args.addScalarBits(scratchPtr, DataType.I64);

        CKernelExecutable exec = (CKernelExecutable) backend.getOrCompile(program, key);
        exec.launch(
                new LaunchConfig(1, 1, 1, 1, 1, 1, 0, false),
                args,
                new ExecutionStream(Device.C, 0L, true));

        return outputViews.getFirst();
    }

    private List<Tensor> resolveInputs(LIRGraph graph, List<Tensor> inputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        List<Tensor> resolved = new ArrayList<>(inputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            Tensor tensor = inputs.get(i);
            if (input instanceof ai.qxotic.jota.ir.lir.ScalarInput) {
                resolved.add(tensor);
                continue;
            }
            MemoryView<?> view = tensor.tryGetMaterialized().orElseGet(tensor::materialize);
            @SuppressWarnings("unchecked")
            MemoryDomain<Object> srcContext =
                    (MemoryDomain<Object>)
                            Environment.current().runtimeFor(view.memory().device()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryView<Object> srcView = (MemoryView<Object>) view;
            MemoryView<Object> contig =
                    MemoryView.of(
                            srcContext
                                    .memoryAllocator()
                                    .allocateMemory(view.dataType(), view.shape()),
                            view.dataType(),
                            Layout.rowMajor(view.shape()));
            MemoryDomain.copy(srcContext, srcView, srcContext, contig);
            view = contig;
            srcView = contig;
            if (view.memory().device().equals(Device.C)
                    && view.memory().base() instanceof MemorySegment) {
                resolved.add(Tensor.of(view));
                continue;
            }
            MemoryView<MemorySegment> copy =
                    MemoryView.of(
                            memoryDomain
                                    .memoryAllocator()
                                    .allocateMemory(view.dataType(), view.shape()),
                            view.dataType(),
                            Layout.rowMajor(view.shape()));
            MemoryDomain.copy(srcContext, srcView, memoryDomain, copy);
            resolved.add(Tensor.of(copy));
        }
        return resolved;
    }

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

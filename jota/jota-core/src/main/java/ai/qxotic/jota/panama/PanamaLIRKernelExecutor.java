package ai.qxotic.jota.panama;

import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRStandardPipeline;
import ai.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.JavaKernel;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PanamaLIRKernelExecutor {

    private final LIRKernelCompiler compiler;
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));

    public PanamaLIRKernelExecutor(DiskKernelCache cache) {
        this.compiler = new LIRKernelCompiler(Objects.requireNonNull(cache, "cache"));
    }

    public record CompiledKernel(JavaKernel kernel, ScratchLayout scratchLayout) {}

    public CompiledKernel compile(LIRGraph graph) {
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(graph, scratchLayout);
        }
        JavaKernel kernel = compiler.compile(graph, scratchLayout);
        return new CompiledKernel(kernel, scratchLayout);
    }

    public MemoryView<?> execute(
            TIRGraph graph, List<Tensor> lirInputs, MemoryDomain<MemorySegment> memoryDomain) {
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph));
        return execute(lirGraph, lirInputs, memoryDomain);
    }

    public void execute(
            LIRGraph graph,
            CompiledKernel compiled,
            List<Tensor> lirInputs,
            List<MemoryView<?>> outputs,
            MemoryDomain<MemorySegment> memoryDomain,
            Memory<MemorySegment> scratch) {
        compiled.kernel()
                .execute(memoryDomain, argsBuilder.build(graph, lirInputs, outputs), scratch);
    }

    public void execute(
            LIRGraph graph,
            CompiledKernel compiled,
            List<MemoryView<?>> inputs,
            List<ai.qxotic.jota.tensor.ScalarArg> scalars,
            List<MemoryView<?>> outputs,
            MemoryDomain<MemorySegment> memoryDomain,
            Memory<MemorySegment> scratch) {
        compiled.kernel()
                .execute(memoryDomain, argsBuilder.build(graph, inputs, scalars, outputs), scratch);
    }

    public MemoryView<?> execute(
            LIRGraph graph, List<Tensor> lirInputs, MemoryDomain<MemorySegment> memoryDomain) {
        // Analyze scratch requirements
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);

        // Verify scratch layout if enabled
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(graph, scratchLayout);
        }

        // Compile kernel with scratch layout
        JavaKernel kernel = compiler.compile(graph, scratchLayout);
        return execute(graph, kernel, scratchLayout, lirInputs, memoryDomain);
    }

    public MemoryView<?> execute(
            LIRGraph graph,
            JavaKernel kernel,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain) {
        // For backwards compatibility - analyze scratch if not provided
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);
        return execute(graph, kernel, scratchLayout, lirInputs, memoryDomain);
    }

    public MemoryView<?> execute(
            LIRGraph graph,
            JavaKernel kernel,
            ScratchLayout scratchLayout,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain) {
        List<MemoryView<?>> outputs = allocateOutputs(graph, memoryDomain);

        // Allocate scratch if needed
        Memory<MemorySegment> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch = allocateScratch(memoryDomain, scratchLayout.alignedTotalByteSize());
        }

        // Execute kernel with scratch
        kernel.execute(memoryDomain, argsBuilder.build(graph, lirInputs, outputs), scratch);

        return outputs.getFirst();
    }

    private List<MemoryView<?>> allocateOutputs(
            LIRGraph graph, MemoryDomain<MemorySegment> memoryDomain) {
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

    private Memory<MemorySegment> allocateScratch(
            MemoryDomain<MemorySegment> memoryDomain, long byteSize) {
        return memoryDomain.memoryAllocator().allocateMemory(byteSize, 64L);
    }
}

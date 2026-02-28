package com.qxotic.jota.runtime.panama;

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
import com.qxotic.jota.tensor.DiskKernelCache;
import com.qxotic.jota.tensor.JavaKernel;
import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelLaunchContext;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class PanamaLIRKernelExecutor {

    private final LIRKernelCompiler compiler;
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final ConcurrentMap<KernelCacheKey, CompiledKernel> compiledKernelCache =
            new ConcurrentHashMap<>();

    public PanamaLIRKernelExecutor(DiskKernelCache cache) {
        this.compiler = new LIRKernelCompiler(Objects.requireNonNull(cache, "cache"));
    }

    public record CompiledKernel(JavaKernel kernel, ScratchLayout scratchLayout) {}

    public CompiledKernel compile(LIRGraph graph) {
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(graph, scratchLayout);
        }
        KernelCacheKey key = compiler.cacheKeyFor(graph, scratchLayout);
        return compiledKernelCache.computeIfAbsent(
                key,
                __ -> {
                    JavaKernel kernel = compiler.compile(graph, scratchLayout);
                    return new CompiledKernel(kernel, scratchLayout);
                });
    }

    public MemoryView<?> execute(
            TIRGraph graph, List<Tensor> lirInputs, MemoryDomain<MemorySegment> memoryDomain) {
        return execute(graph, lirInputs, memoryDomain, KernelLaunchContext.disabled());
    }

    public MemoryView<?> execute(
            TIRGraph graph,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain,
            KernelLaunchContext launchContext) {
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph));
        return execute(lirGraph, lirInputs, memoryDomain, launchContext);
    }

    public void execute(
            LIRGraph graph,
            CompiledKernel compiled,
            List<Tensor> lirInputs,
            List<MemoryView<?>> outputs,
            MemoryDomain<MemorySegment> memoryDomain,
            Memory<MemorySegment> scratch,
            KernelLaunchContext launchContext) {
        compiled.kernel()
                .execute(
                        memoryDomain,
                        argsBuilder.build(graph, lirInputs, outputs),
                        scratch,
                        launchContext);
    }

    public void execute(
            LIRGraph graph,
            CompiledKernel compiled,
            List<MemoryView<?>> inputs,
            List<com.qxotic.jota.tensor.ScalarArg> scalars,
            List<MemoryView<?>> outputs,
            MemoryDomain<MemorySegment> memoryDomain,
            Memory<MemorySegment> scratch,
            KernelLaunchContext launchContext) {
        compiled.kernel()
                .execute(
                        memoryDomain,
                        argsBuilder.build(graph, inputs, scalars, outputs),
                        scratch,
                        launchContext);
    }

    public MemoryView<?> execute(
            LIRGraph graph, List<Tensor> lirInputs, MemoryDomain<MemorySegment> memoryDomain) {
        return execute(graph, lirInputs, memoryDomain, KernelLaunchContext.disabled());
    }

    public MemoryView<?> execute(
            LIRGraph graph,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain,
            KernelLaunchContext launchContext) {
        CompiledKernel compiled = compile(graph);
        return execute(
                graph,
                compiled.kernel(),
                compiled.scratchLayout(),
                lirInputs,
                memoryDomain,
                launchContext);
    }

    public MemoryView<?> execute(
            LIRGraph graph,
            JavaKernel kernel,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain) {
        // For backwards compatibility - analyze scratch if not provided
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);
        return execute(
                graph,
                kernel,
                scratchLayout,
                lirInputs,
                memoryDomain,
                KernelLaunchContext.disabled());
    }

    public MemoryView<?> execute(
            LIRGraph graph,
            JavaKernel kernel,
            ScratchLayout scratchLayout,
            List<Tensor> lirInputs,
            MemoryDomain<MemorySegment> memoryDomain,
            KernelLaunchContext launchContext) {
        List<MemoryView<?>> outputs = allocateOutputs(graph, memoryDomain);

        // Allocate scratch if needed
        Memory<MemorySegment> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch = allocateScratch(memoryDomain, scratchLayout.alignedTotalByteSize());
        }

        // Execute kernel with scratch
        kernel.execute(
                memoryDomain, argsBuilder.build(graph, lirInputs, outputs), scratch, launchContext);

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

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
import com.qxotic.jota.runtime.DiskKernelCache;
import com.qxotic.jota.runtime.JavaKernel;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.clike.LIRKernelArgsBuilder;
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
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final ConcurrentMap<KernelCacheKey, CompiledKernel> compiledKernelCache =
            new ConcurrentHashMap<>();

    public PanamaLIRKernelExecutor(DiskKernelCache cache) {
        this.compiler = new LIRKernelCompiler(Objects.requireNonNull(cache, "cache"));
    }

    private record CompiledKernel(JavaKernel kernel, ScratchLayout scratchLayout) {}

    private CompiledKernel compile(LIRGraph graph) {
        ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
        ScratchLayout scratchLayout = scratchAnalysis.analyze(graph);
        if (verifyScratch) {
            ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
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
        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRStandardPipeline pipeline = new LIRStandardPipeline();
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph));
        return execute(lirGraph, lirInputs, memoryDomain);
    }

    public MemoryView<?> execute(
            LIRGraph graph, List<Tensor> lirInputs, MemoryDomain<MemorySegment> memoryDomain) {
        CompiledKernel compiled = compile(graph);
        List<MemoryView<?>> outputs = allocateOutputs(graph, memoryDomain);

        Memory<MemorySegment> scratch = null;
        if (compiled.scratchLayout().requiresScratch()) {
            scratch =
                    allocateScratch(memoryDomain, compiled.scratchLayout().alignedTotalByteSize());
        }

        compiled.kernel()
                .execute(memoryDomain, argsBuilder.build(graph, lirInputs, outputs), scratch);

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

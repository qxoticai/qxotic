package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.clike.CLikeDialect;

final class MetalDialect implements CLikeDialect {

    @Override
    public String language() {
        return "metal";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return MetalKernelProgramGenerator.SourceGenerator.generate(
                graph, scratchLayout, kernelName);
    }
}

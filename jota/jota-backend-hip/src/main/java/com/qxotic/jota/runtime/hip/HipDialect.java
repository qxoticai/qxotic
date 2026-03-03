package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.clike.CLikeDialect;

final class HipDialect implements CLikeDialect {

    @Override
    public String language() {
        return "hip";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return HipKernelProgramGenerator.SourceGenerator.generate(graph, scratchLayout, kernelName);
    }
}

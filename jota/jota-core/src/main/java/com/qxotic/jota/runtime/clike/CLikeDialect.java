package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;

public interface CLikeDialect {
    String language();

    String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName);
}

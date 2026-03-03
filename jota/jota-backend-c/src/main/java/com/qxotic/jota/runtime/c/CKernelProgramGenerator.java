package com.qxotic.jota.runtime.c;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.clike.CLikeKernelGenerator;

final class CKernelProgramGenerator {

    private final CLikeKernelGenerator generator = new CLikeKernelGenerator(new CDialect());

    KernelProgram generate(LIRGraph graph, ScratchLayout scratchLayout, KernelCacheKey key) {
        return generator.generate(graph, scratchLayout, key);
    }
}

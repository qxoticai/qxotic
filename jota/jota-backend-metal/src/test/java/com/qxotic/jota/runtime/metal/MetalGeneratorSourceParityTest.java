package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.runtime.clike.CLikeParityTestSupport;
import org.junit.jupiter.api.Test;

class MetalGeneratorSourceParityTest {

    @Test
    void boolLessThanFp64StaysInFloatingDomain() {
        String source =
                CLikeParityTestSupport.renderBinaryScalarSource(
                        (graph, kernelName) ->
                                MetalKernelProgramGenerator.SourceGenerator.generate(
                                        graph, ScratchLayout.EMPTY, kernelName),
                        "parity_kernel_metal",
                        BinaryOperator.LESS_THAN,
                        DataType.BOOL,
                        DataType.FP64,
                        DataType.BOOL);
        CLikeParityTestSupport.assertBoolLessThanFp64InFloatingDomain(
                source, "1.0 : 0.0", "(int)(scalar0)");
    }
}

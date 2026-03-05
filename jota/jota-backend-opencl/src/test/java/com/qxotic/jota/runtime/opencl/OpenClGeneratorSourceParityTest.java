package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.runtime.clike.CLikeParityTestSupport;
import org.junit.jupiter.api.Test;

class OpenClGeneratorSourceParityTest {

    @Test
    void boolLessThanFp64StaysInFloatingDomain() {
        String source =
                CLikeParityTestSupport.renderBinaryScalarSource(
                        (graph, kernelName) ->
                                new OpenClDialect()
                                        .renderSource(graph, ScratchLayout.EMPTY, kernelName),
                        "parity_kernel_opencl",
                        BinaryOperator.LESS_THAN,
                        DataType.BOOL,
                        DataType.FP64,
                        DataType.BOOL);
        CLikeParityTestSupport.assertBoolLessThanFp64InFloatingDomain(
                source, "1.0 : 0.0", "(int)(scalar0)");
    }
}

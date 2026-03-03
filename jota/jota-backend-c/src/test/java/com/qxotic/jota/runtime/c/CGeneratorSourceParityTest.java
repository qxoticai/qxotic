package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRExprGraph;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import java.util.List;
import org.junit.jupiter.api.Test;

class CGeneratorSourceParityTest {

    @Test
    void boolAddUsesNumericNormalization() {
        String source =
                generateBinaryKernelSource(
                        BinaryOperator.ADD, DataType.BOOL, DataType.I32, DataType.I32);
        assertTrue(source.contains("? 1 : 0"));
    }

    @Test
    void boolLessThanFp64StaysInFloatingDomain() {
        String source =
                generateBinaryKernelSource(
                        BinaryOperator.LESS_THAN, DataType.BOOL, DataType.FP64, DataType.BOOL);
        assertTrue(source.contains("1.0 : 0.0"));
        assertFalse(source.contains("(int32_t)(scalar0)"));
    }

    private static String generateBinaryKernelSource(
            BinaryOperator op, DataType leftType, DataType rightType, DataType outType) {
        LIRGraph graph = buildBinaryScalarGraph(op, leftType, rightType, outType);
        return new CDialect().renderSource(graph, ScratchLayout.EMPTY, "parity_kernel_c");
    }

    private static LIRGraph buildBinaryScalarGraph(
            BinaryOperator op, DataType leftType, DataType rightType, DataType outType) {
        LIRGraph.Builder builder = LIRGraph.builder();
        var leftInput = builder.addScalarInput(leftType);
        var rightInput = builder.addScalarInput(rightType);
        BufferRef out = builder.addContiguousOutput(outType, 1);

        LIRExprGraph expr = builder.exprGraph();
        LIRExprNode left = expr.scalarInput(leftInput.id(), leftType);
        LIRExprNode right = expr.scalarInput(rightInput.id(), rightType);
        LIRExprNode binary = expr.scalarBinary(op, left, right);
        LIRExprNode store = expr.store(out, expr.indexConst(0), binary);
        LIRExprNode root = expr.block(List.of(store));
        return builder.build(root);
    }
}

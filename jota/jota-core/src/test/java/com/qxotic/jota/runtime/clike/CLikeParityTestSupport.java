package com.qxotic.jota.runtime.clike;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRExprGraph;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.tir.BinaryOperator;
import java.util.List;

public final class CLikeParityTestSupport {
    private CLikeParityTestSupport() {}

    @FunctionalInterface
    public interface SourceRenderer {
        String render(LIRGraph graph, String kernelName);
    }

    public static LIRGraph buildBinaryScalarGraph(
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
        return builder.build(expr.block(List.of(store)));
    }

    public static String renderBinaryScalarSource(
            SourceRenderer renderer,
            String kernelName,
            BinaryOperator op,
            DataType leftType,
            DataType rightType,
            DataType outType) {
        LIRGraph graph = buildBinaryScalarGraph(op, leftType, rightType, outType);
        return renderer.render(graph, kernelName);
    }

    public static void assertBoolLessThanFp64InFloatingDomain(
            String source, String expectedFloatBoolPattern, String forbiddenBoolCastPattern) {
        assertTrue(source.contains(expectedFloatBoolPattern));
        assertFalse(source.contains(forbiddenBoolCastPattern));
    }
}

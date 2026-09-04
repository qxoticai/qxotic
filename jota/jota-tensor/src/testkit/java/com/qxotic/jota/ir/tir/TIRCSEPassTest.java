package com.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Tests for TIR Common Subexpression Elimination (CSE) pass. */
class TIRCSEPassTest {

    @Test
    void testSimpleCSE() {
        // Build graph: a = x * y; b = x * y (should share the multiply)
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);

        BinaryOp mul1 = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp mul2 = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x, y), List.of(mul1, mul2));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        // Both outputs should be the same node after CSE
        assertSame(
                optimized.outputs().get(0),
                optimized.outputs().get(1),
                "Duplicate multiply operations should be shared");
    }

    @Test
    void testNoCSEForDifferentOps() {
        // Build graph: a = x * y; b = x + y (should NOT share)
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);

        BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp add = new BinaryOp(BinaryOperator.ADD, x, y, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x, y), List.of(mul, add));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        // Outputs should be different nodes
        assertNotSame(
                optimized.outputs().get(0),
                optimized.outputs().get(1),
                "Different operations should not be shared");
    }

    @Test
    void testNoCSEForDifferentInputs() {
        // Build graph: a = x * y; b = x * z (should NOT share)
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);
        TensorInput z = new TensorInput(2, DataType.FP32, layout);

        BinaryOp mul1 = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp mul2 = new BinaryOp(BinaryOperator.MULTIPLY, x, z, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x, y, z), List.of(mul1, mul2));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        // Outputs should be different nodes
        assertNotSame(
                optimized.outputs().get(0),
                optimized.outputs().get(1),
                "Operations with different inputs should not be shared");
    }

    @Test
    void testCSEWithUnaryOps() {
        // Build graph: a = neg(x); b = neg(x) (should share)
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);

        UnaryOp neg1 = new UnaryOp(UnaryOperator.NEGATE, x, layout.shape());
        UnaryOp neg2 = new UnaryOp(UnaryOperator.NEGATE, x, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x), List.of(neg1, neg2));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        assertSame(
                optimized.outputs().get(0),
                optimized.outputs().get(1),
                "Duplicate unary operations should be shared");
    }

    @Test
    void testCSEInSubexpression() {
        // Build graph: a = (x * y) + z; b = (x * y) - z
        // The x*y should be shared
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);
        TensorInput z = new TensorInput(2, DataType.FP32, layout);

        BinaryOp mul1 = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp add = new BinaryOp(BinaryOperator.ADD, mul1, z, layout.shape());

        BinaryOp mul2 = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp sub = new BinaryOp(BinaryOperator.SUBTRACT, mul2, z, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x, y, z), List.of(add, sub));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        // The multiply operations should be shared
        BinaryOp optAdd = (BinaryOp) optimized.outputs().get(0);
        BinaryOp optSub = (BinaryOp) optimized.outputs().get(1);

        assertSame(optAdd.left(), optSub.left(), "Common subexpression (x * y) should be shared");
    }

    @Test
    void testCSEPreservesGraphStructure() {
        // Ensure CSE doesn't break the graph structure
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);

        BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        UnaryOp neg = new UnaryOp(UnaryOperator.NEGATE, mul, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x, y), List.of(neg));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        assertEquals(1, optimized.outputs().size());
        assertTrue(optimized.outputs().get(0) instanceof UnaryOp);

        UnaryOp optNeg = (UnaryOp) optimized.outputs().get(0);
        assertTrue(optNeg.input() instanceof BinaryOp);
    }

    @Test
    void testCSEWithConstants() {
        // Constants should not be CSE'd (they're leaves)
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);

        ScalarConstant two1 =
                ScalarConstant.broadcast(
                        Float.floatToRawIntBits(2.0f), DataType.FP32, layout.shape());
        ScalarConstant two2 =
                ScalarConstant.broadcast(
                        Float.floatToRawIntBits(2.0f), DataType.FP32, layout.shape());

        BinaryOp mul1 = new BinaryOp(BinaryOperator.MULTIPLY, x, two1, layout.shape());
        BinaryOp mul2 = new BinaryOp(BinaryOperator.MULTIPLY, x, two2, layout.shape());

        TIRGraph graph = new TIRGraph(List.of(x), List.of(mul1, mul2));

        TIRCSEPass cse = new TIRCSEPass();
        TIRGraph optimized = cse.run(graph);

        // The multiplies should be shared (same input x and equivalent constants)
        assertSame(
                optimized.outputs().get(0),
                optimized.outputs().get(1),
                "Operations with equivalent constant inputs should be shared");
    }
}

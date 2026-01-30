package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import org.junit.jupiter.api.Test;

class LIRTextRendererTest {

    @Test
    void testSimpleElementwiseAdd() {
        // Build a simple elementwise add graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.mul(i, new IndexConst(4)); // 4 = Float.BYTES
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        ScalarExpr v1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, v0, v1);
        Store store = new Store(out, offset, sum);

        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Print for debugging
        System.out.println("=== LIR Output ===");
        System.out.println(output);
        System.out.println("==================");

        // Verify key elements are present (note: DataType uses lowercase like fp32)
        assertTrue(output.contains("LIRGraph {"), "Should contain 'LIRGraph {'");
        assertTrue(output.contains("inputs:"), "Should contain 'inputs:'");
        assertTrue(output.contains("outputs:"), "Should contain 'outputs:'");
        assertTrue(output.contains("body {"), "Should contain 'body {'");
        assertTrue(
                output.contains("parallel.for %i in [0, 4) {"),
                "Should contain 'parallel.for %i in [0, 4) {'");
        assertTrue(output.contains("load fp32 %in0"), "Should contain 'load fp32 %in0'");
        assertTrue(output.contains("load fp32 %in1"), "Should contain 'load fp32 %in1'");
        assertTrue(output.contains("add fp32"), "Should contain 'add fp32'");
        assertTrue(output.contains("store %out0"), "Should contain 'store %out0'");
    }

    @Test
    void testUnaryNegate() {
        // Build a unary negate graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.mul(i, new IndexConst(4));
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        ScalarExpr neg = new ScalarUnary(UnaryOperator.NEGATE, v0);
        Store store = new Store(out, offset, neg);

        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        assertTrue(output.contains("neg fp32"));
    }

    @Test
    void testBufferMetadata() {
        // Build a graph to check buffer metadata formatting
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 2, 3);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 2, 3);

        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexExpr linearIdx = IndexBinary.add(IndexBinary.mul(i, new IndexConst(3)), j);
        IndexExpr offset = IndexBinary.mul(linearIdx, new IndexConst(4));
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, v0);

        Loop innerLoop = Loop.parallel("j", 3, store);
        Loop outerLoop = Loop.parallel("i", 2, innerLoop);
        LIRGraph graph = builder.build(outerLoop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Check buffer metadata includes shape and strides with new format
        assertTrue(output.contains("fp32[(2, 3):(12, 4)] contiguous"));
    }

    @Test
    void testGelu() {
        // Build GELU graph: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 5);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 5);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.mul(i, new IndexConst(4));
        ScalarExpr x = new ScalarLoad(in0, offset);

        // Constants
        ScalarExpr c_0_044715 = ScalarConst.ofFloat(0.044715f);
        ScalarExpr c_sqrt_2_pi = ScalarConst.ofFloat(0.79788456f);
        ScalarExpr c_1 = ScalarConst.ofFloat(1.0f);
        ScalarExpr c_0_5 = ScalarConst.ofFloat(0.5f);

        // Build GELU expression tree
        ScalarExpr x_squared = new ScalarBinary(BinaryOperator.MULTIPLY, x, x);
        ScalarExpr x_cubed = new ScalarBinary(BinaryOperator.MULTIPLY, x_squared, x);
        ScalarExpr scaled_cubic = new ScalarBinary(BinaryOperator.MULTIPLY, c_0_044715, x_cubed);
        ScalarExpr inner_sum = new ScalarBinary(BinaryOperator.ADD, x, scaled_cubic);
        ScalarExpr scaled_inner = new ScalarBinary(BinaryOperator.MULTIPLY, c_sqrt_2_pi, inner_sum);
        ScalarExpr tanh_result = new ScalarUnary(UnaryOperator.TANH, scaled_inner);
        ScalarExpr one_plus_tanh = new ScalarBinary(BinaryOperator.ADD, c_1, tanh_result);
        ScalarExpr x_times_bracket = new ScalarBinary(BinaryOperator.MULTIPLY, x, one_plus_tanh);
        ScalarExpr gelu = new ScalarBinary(BinaryOperator.MULTIPLY, c_0_5, x_times_bracket);

        Store store = new Store(out, offset, gelu);
        Loop loop = Loop.parallel("i", 5, store);
        LIRGraph graph = builder.build(loop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Print for debugging
        System.out.println("=== GELU LIR Output ===");
        System.out.println(output);
        System.out.println("=======================");

        // Verify key elements are present
        assertTrue(
                output.contains("parallel.for %i in [0, 5)"),
                "Should contain loop with correct bounds");
        assertTrue(output.contains("0.044715f"), "Should contain cubic coefficient constant");
        assertTrue(output.contains("0.7978846f"), "Should contain sqrt(2/pi) constant");
        assertTrue(output.contains("tanh fp32"), "Should contain tanh operation");
        assertTrue(output.contains("mul fp32"), "Should contain multiply operations");
        assertTrue(output.contains("add fp32"), "Should contain add operations");
        assertTrue(output.contains("store %out0"), "Should store to output");
    }
}

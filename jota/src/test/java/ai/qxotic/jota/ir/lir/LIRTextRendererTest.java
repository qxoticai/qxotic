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
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4)); // 4 = Float.BYTES
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        ScalarExpr v1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, v0, v1);
        Store store = new Store(out, offset, sum);

        StructuredFor loop =
                new StructuredFor(
                        "i",
                        new IndexConst(0),
                        new IndexConst(4),
                        new IndexConst(1),
                        java.util.List.of(),
                        new Block(java.util.List.of(store, Yield.empty())));
        LIRGraph graph = builder.build(loop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Print for debugging
        System.out.println("=== LIR Output ===");
        System.out.println(output);
        System.out.println("==================");

        // Verify MLIR-style output
        assertTrue(output.contains("module {"), "Should contain 'module {'");
        assertTrue(output.contains("func.func @kernel"), "Should contain function definition");
        assertTrue(output.contains("%in0: memref<4xfp32>"), "Should contain input memref type");
        assertTrue(output.contains("%out0: memref<4xfp32>"), "Should contain output memref type");
        assertTrue(output.contains("scf.for %i = 0 to 4 step 1"), "Should contain scf.for loop");
        assertTrue(output.contains("memref.load %in0[%i * 4]"), "Should contain memref.load");
        assertTrue(output.contains("arith.addf"), "Should contain arith.addf");
        assertTrue(output.contains("memref.store"), "Should contain memref.store");
    }

    @Test
    void testUnaryNegate() {
        // Build a unary negate graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        ScalarExpr neg = new ScalarUnary(UnaryOperator.NEGATE, v0);
        Store store = new Store(out, offset, neg);

        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Verify MLIR operator name
        assertTrue(output.contains("arith.negf"), "Should contain 'arith.negf'");
    }

    @Test
    void testBufferMetadata() {
        // Build a graph to check buffer metadata formatting
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 2, 3);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 2, 3);

        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexExpr linearIdx = IndexBinary.add(IndexBinary.multiply(i, new IndexConst(3)), j);
        IndexExpr offset = IndexBinary.multiply(linearIdx, new IndexConst(4));
        ScalarExpr v0 = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, v0);

        Loop innerLoop = Loop.parallel("j", 3, store);
        Loop outerLoop = Loop.parallel("i", 2, innerLoop);
        LIRGraph graph = builder.build(outerLoop);

        // Render
        LIRTextRenderer renderer = new LIRTextRenderer();
        String output = renderer.render(graph);

        // Print for debugging
        System.out.println("=== LIR Output (testBufferMetadata) ===");
        System.out.println(output);
        System.out.println("========================================");

        // Verify MLIR memref type format
        assertTrue(output.contains("memref<2x3xfp32>"), "Should contain memref type with shape");
        assertTrue(output.contains("scf.parallel"), "Should contain scf.parallel loop");
        assertTrue(output.contains(" * "), "Should contain index multiplication operators");
        assertTrue(output.contains(" + "), "Should contain index addition operators");
        assertTrue(output.contains("memref.store"), "Should store to output");
    }

    //    @Test
    //    void testGelu() {
    //        // Build GELU graph: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    //        LIRGraph.Builder builder = LIRGraph.builder();
    //        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 5);
    //        BufferRef out = builder.addContiguousOutput(DataType.FP32, 5);
    //
    //        IndexVar i = new IndexVar("i");
    //        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
    //        ScalarExpr x = new ScalarLoad(in0, offset);
    //
    //        // Constants
    //        ScalarExpr c_0_044715 = ScalarLiteral.ofFloat(0.044715f);
    //        ScalarExpr c_sqrt_2_pi = ScalarLiteral.ofFloat(0.79788456f);
    //        ScalarExpr c_1 = ScalarLiteral.ofFloat(1.0f);
    //        ScalarExpr c_0_5 = ScalarLiteral.ofFloat(0.5f);
    //
    //        // Build GELU expression tree
    //        ScalarExpr x_squared = new ScalarBinary(BinaryOperator.MULTIPLY, x, x);
    //        ScalarExpr x_cubed = new ScalarBinary(BinaryOperator.MULTIPLY, x_squared, x);
    //        ScalarExpr scaled_cubic = new ScalarBinary(BinaryOperator.MULTIPLY, c_0_044715,
    // x_cubed);
    //        ScalarExpr inner_sum = new ScalarBinary(BinaryOperator.ADD, x, scaled_cubic);
    //        ScalarExpr scaled_inner = new ScalarBinary(BinaryOperator.MULTIPLY, c_sqrt_2_pi,
    // inner_sum);
    //        ScalarExpr tanh_result = new ScalarUnary(UnaryOperator.TANH, scaled_inner);
    //        ScalarExpr one_plus_tanh = new ScalarBinary(BinaryOperator.ADD, c_1, tanh_result);
    //        ScalarExpr x_times_bracket = new ScalarBinary(BinaryOperator.MULTIPLY, x,
    // one_plus_tanh);
    //        ScalarExpr gelu = new ScalarBinary(BinaryOperator.MULTIPLY, c_0_5, x_times_bracket);
    //
    //        Store store = new Store(out, offset, gelu);
    //        Loop loop = Loop.parallel("i", 5, store);
    //        LIRGraph graph = builder.build(loop);
    //
    //        // Render
    //        LIRTextRenderer renderer = new LIRTextRenderer();
    //        String output = renderer.render(graph);
    //
    //        // Print for debugging
    //        System.out.println("=== GELU LIR Output ===");
    //        System.out.println(output);
    //        System.out.println("=======================");

    @Test
    void testGelu() {
        // Build GELU graph: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 5);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 5);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr x = new ScalarLoad(in0, offset);

        // Constants
        ScalarExpr c_0_044715 = ScalarLiteral.ofFloat(0.044715f);
        ScalarExpr c_sqrt_2_pi = ScalarLiteral.ofFloat(0.79788456f);
        ScalarExpr c_1 = ScalarLiteral.ofFloat(1.0f);
        ScalarExpr c_0_5 = ScalarLiteral.ofFloat(0.5f);

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

        // Verify MLIR-style output
        assertTrue(
                output.contains("scf.parallel (%i) = (0) to (5) step (1)"),
                "Should contain scf.parallel loop");
        assertTrue(output.contains("0.044715f"), "Should contain cubic coefficient constant");
        assertTrue(output.contains("0.7978846f"), "Should contain sqrt(2/pi) constant");
        assertTrue(output.contains("math.tanh"), "Should contain math.tanh operation");
        assertTrue(output.contains("arith.mulf"), "Should contain arith.mulf operations");
        assertTrue(output.contains("arith.addf"), "Should contain arith.addf operations");
        assertTrue(output.contains("memref.store"), "Should store to output");
    }
}

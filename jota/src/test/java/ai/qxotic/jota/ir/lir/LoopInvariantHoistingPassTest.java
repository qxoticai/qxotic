package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class LoopInvariantHoistingPassTest {

    @Test
    void testHoistLoopInvariantMultiplication() {
        // Build: for i in [0, 4) { store out[i], (scalar0 * scalar1) }
        // The multiplication is loop-invariant and should be hoisted
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr product = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        Store store = new Store(out, offset, product);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original LIR ===");
        System.out.println(original);

        // Apply hoisting pass
        LoopInvariantHoistingPass hoistingPass = new LoopInvariantHoistingPass();
        LIRGraph hoisted = hoistingPass.run(graph);

        // Print hoisted
        String hoistedStr = renderer.render(hoisted);
        System.out.println("=== Hoisted LIR ===");
        System.out.println(hoistedStr);

        // Verify hoisting occurred
        assertTrue(hoistedStr.contains("%hoisted0"), "Should have hoisted expression");
        assertTrue(hoistedStr.contains("multiply fp32"), "Should contain multiply");
        assertTrue(hoistedStr.contains("%hoisted0"), "Should reference hoisted value");

        // Execute and verify correct result
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(16);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(2.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32);
            interpreter.bindBuffer(2, output);
            interpreter.execute(hoisted);

            // All elements should be 2.0 * 3.0 = 6.0
            for (int j = 0; j < 4; j++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                assertEquals(6.0f, actual, 0.0001f, "Element " + j);
            }
        }
    }

    @Test
    void testHoistInNestedLoop() {
        // Build: for i in [0, 2) { for j in [0, 3) { store out[i*3+j], (scalar0 * scalar1) } }
        // The multiplication is invariant to both loops
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 2, 3);

        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexExpr linearIdx = IndexBinary.add(IndexBinary.multiply(i, new IndexConst(3)), j);
        IndexExpr offset = IndexBinary.multiply(linearIdx, new IndexConst(4));
        ScalarExpr product = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        Store store = new Store(out, offset, product);

        Loop innerLoop = Loop.parallel("j", 3, store);
        Loop outerLoop = Loop.parallel("i", 2, innerLoop);

        LIRGraph graph = builder.build(outerLoop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        System.out.println("=== Original Nested LIR ===");
        System.out.println(renderer.render(graph));

        // Apply hoisting pass
        LoopInvariantHoistingPass hoistingPass = new LoopInvariantHoistingPass();
        LIRGraph hoisted = hoistingPass.run(graph);

        // Print hoisted
        String hoistedStr = renderer.render(hoisted);
        System.out.println("=== Hoisted Nested LIR ===");
        System.out.println(hoistedStr);

        // Verify hoisting occurred - should be hoisted out of BOTH loops (not just inner)
        assertTrue(hoistedStr.contains("%hoisted"), "Should have hoisted expression");
        // The multiply should appear BEFORE the outer loop, not inside it
        int letPos = hoistedStr.indexOf("%hoisted");
        int outerLoopPos = hoistedStr.indexOf("parallel.for %i");
        assertTrue(letPos < outerLoopPos, "Hoisted expression should be outside outer loop");

        // Execute and verify
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(24);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(5.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(7.0f), DataType.FP32);
            interpreter.bindBuffer(2, output);
            interpreter.execute(hoisted);

            // All elements should be 5.0 * 7.0 = 35.0
            for (int k = 0; k < 6; k++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, k);
                assertEquals(35.0f, actual, 0.0001f, "Element " + k);
            }
        }
    }

    @Test
    void testNoHoistingWhenLoopDependent() {
        // Build: for i in [0, 4) { store out[i], (scalar0 * load(in1[i])) }
        // The load depends on loop index, so multiplication cannot be hoisted
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr load = new ScalarLoad(in1, offset);
        ScalarExpr product = new ScalarBinary(BinaryOperator.MULTIPLY, in0, load);
        Store store = new Store(out, offset, product);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Apply hoisting pass
        LoopInvariantHoistingPass hoistingPass = new LoopInvariantHoistingPass();
        LIRGraph hoisted = hoistingPass.run(graph);

        // Print result
        LIRTextRenderer renderer = new LIRTextRenderer();
        String hoistedStr = renderer.render(hoisted);
        System.out.println("=== No Hoisting Expected ===");
        System.out.println(hoistedStr);

        // Should NOT have any let bindings since multiplication depends on loop
        assertFalse(hoistedStr.contains("%hoisted"), "Should not hoist loop-dependent expression");
    }

    @Test
    void testHoistWithReduction() {
        // Build the typical pattern from matmul:
        // for i in [0, 3) iter_args(%acc = 0) { yield add %acc, (scalar0 * scalar1) }
        // The multiplication should be hoisted outside the loop
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarExpr product = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);

        ScalarExpr init = ScalarLiteral.ofFloat(0.0f);
        ScalarExpr updated =
                new ScalarBinary(
                        BinaryOperator.ADD, new ScalarRef("acc", DataType.FP32), product);
        StructuredFor loop =
                new StructuredFor(
                        "k",
                        new IndexConst(0),
                        new IndexConst(3),
                        new IndexConst(1),
                        java.util.List.of(new LoopIterArg("acc", DataType.FP32, init)),
                        new Yield(java.util.List.of(updated)));

        Store store = new Store(out, new IndexConst(0), new ScalarRef("acc", DataType.FP32));
        Block body = new Block(java.util.List.of(loop, store));

        LIRGraph graph = builder.build(body);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        System.out.println("=== Original Reduction LIR ===");
        System.out.println(renderer.render(graph));

        // Apply hoisting pass
        LoopInvariantHoistingPass hoistingPass = new LoopInvariantHoistingPass();
        LIRGraph hoisted = hoistingPass.run(graph);

        // Print hoisted
        String hoistedStr = renderer.render(hoisted);
        System.out.println("=== Hoisted Reduction LIR ===");
        System.out.println(hoistedStr);

        // Verify hoisting
        assertTrue(hoistedStr.contains("%hoisted0"), "Should have hoisted multiplication");

        // Execute and verify: 3 iterations of adding (2*3=6), starting from 0
        // Result should be 0 + 6 + 6 + 6 = 18
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(4);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(2.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32);
            interpreter.bindBuffer(2, output);
            interpreter.execute(hoisted);

            float result = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(18.0f, result, 0.0001f);
        }
    }
}

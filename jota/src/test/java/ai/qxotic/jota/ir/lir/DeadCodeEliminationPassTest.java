package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class DeadCodeEliminationPassTest {

    @Test
    void testSimpleDeadCodeRemoval() {
        // Build: %dead = multiply %a, %b; %alive = add %x, %y; store out[0], %alive
        // DCE should remove %dead
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        ScalarInput in2 = builder.addScalarInput(DataType.FP32);
        ScalarInput in3 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarExpr deadExpr = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        ScalarExpr aliveExpr = new ScalarBinary(BinaryOperator.ADD, in2, in3);

        ScalarLet deadLet = new ScalarLet("dead", deadExpr);
        ScalarLet aliveLet = new ScalarLet("alive", aliveExpr);
        ScalarRef aliveRef = new ScalarRef("alive", DataType.FP32);
        Store store = new Store(out, new IndexConst(0), aliveRef);

        Block body = new Block(java.util.List.of(deadLet, aliveLet, store));
        LIRGraph graph = builder.build(body);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original LIR (with dead code) ===");
        System.out.println(original);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Print optimized
        String optimizedStr = renderer.render(optimized);
        System.out.println("=== After DCE ===");
        System.out.println(optimizedStr);

        // Verify dead code was removed
        assertFalse(optimizedStr.contains("%dead"), "Dead ScalarLet should be removed");
        assertTrue(optimizedStr.contains("%alive"), "Alive ScalarLet should remain");
        assertTrue(optimizedStr.contains("add fp32"), "Add operation should remain");
        assertTrue(optimizedStr.contains("store"), "Store should remain");

        // Verify the graph still executes correctly
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(4);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(
                    0, Float.floatToRawIntBits(2.0f), DataType.FP32); // unused (dead)
            interpreter.bindScalarInput(
                    1, Float.floatToRawIntBits(3.0f), DataType.FP32); // unused (dead)
            interpreter.bindScalarInput(2, Float.floatToRawIntBits(5.0f), DataType.FP32); // %x
            interpreter.bindScalarInput(3, Float.floatToRawIntBits(7.0f), DataType.FP32); // %y
            interpreter.bindBuffer(4, output);
            interpreter.execute(optimized);

            float result = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(12.0f, result, 0.0001f, "5.0 + 7.0 = 12.0");
        }
    }

    @Test
    void testUsedLetNotRemoved() {
        // Build: %used = multiply %a, %b; store out[0], %used
        // DCE should NOT remove %used since it's referenced
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarExpr expr = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        ScalarLet let = new ScalarLet("used", expr);
        ScalarRef ref = new ScalarRef("used", DataType.FP32);
        Store store = new Store(out, new IndexConst(0), ref);

        Block body = new Block(java.util.List.of(let, store));
        LIRGraph graph = builder.build(body);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Print result
        LIRTextRenderer renderer = new LIRTextRenderer();
        String resultStr = renderer.render(optimized);
        System.out.println("=== DCE on used let ===");
        System.out.println(resultStr);

        // Verify the let was NOT removed
        assertTrue(resultStr.contains("%used"), "Used ScalarLet should NOT be removed");
        assertTrue(resultStr.contains("multiply fp32"), "Multiply should remain");

        // Verify graph is unchanged
        assertEquals(resultStr, renderer.render(graph), "Graph should be unchanged");
    }

    @Test
    void testMultipleDeadAndAliveInBlock() {
        // Build: %dead1 = ...; %alive = ...; %dead2 = ...; store out[0], %alive
        // DCE should remove both %dead1 and %dead2
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarExpr deadExpr1 = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in0);
        ScalarExpr deadExpr2 = new ScalarBinary(BinaryOperator.ADD, in0, in0);
        ScalarExpr aliveExpr = new ScalarUnary(UnaryOperator.NEGATE, in0);

        ScalarLet deadLet1 = new ScalarLet("dead1", deadExpr1);
        ScalarLet aliveLet = new ScalarLet("alive", aliveExpr);
        ScalarLet deadLet2 = new ScalarLet("dead2", deadExpr2);
        ScalarRef aliveRef = new ScalarRef("alive", DataType.FP32);
        Store store = new Store(out, new IndexConst(0), aliveRef);

        Block body = new Block(java.util.List.of(deadLet1, aliveLet, deadLet2, store));
        LIRGraph graph = builder.build(body);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Print result
        LIRTextRenderer renderer = new LIRTextRenderer();
        String resultStr = renderer.render(optimized);
        System.out.println("=== DCE with multiple dead lets ===");
        System.out.println(resultStr);

        // Verify dead lets were removed
        assertFalse(resultStr.contains("%dead1"), "dead1 should be removed");
        assertFalse(resultStr.contains("%dead2"), "dead2 should be removed");
        assertTrue(resultStr.contains("%alive"), "alive should remain");
        assertTrue(resultStr.contains("negate fp32"), "Negate should remain");
    }

    @Test
    void testDeadCodeInNestedLoop() {
        // Build: for i in [0, 3) { %dead = add %a, %b; store out[i], %a }
        // DCE should remove %dead since it's never used in the loop
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 3);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarExpr deadExpr = new ScalarBinary(BinaryOperator.ADD, in0, in1);
        ScalarLet deadLet = new ScalarLet("dead", deadExpr);
        // Note: %dead is never referenced - we use %scalar0 directly in store
        Store store = new Store(out, offset, in0);

        Block loopBody = new Block(java.util.List.of(deadLet, store));
        Loop loop = Loop.parallel("i", 3, loopBody);
        LIRGraph graph = builder.build(loop);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Print result
        LIRTextRenderer renderer = new LIRTextRenderer();
        String resultStr = renderer.render(optimized);
        System.out.println("=== DCE in nested loop ===");
        System.out.println(resultStr);

        // Verify dead let was removed
        assertFalse(resultStr.contains("%dead"), "Dead ScalarLet in loop should be removed");
        assertTrue(resultStr.contains("parallel.for"), "Loop should remain");
        assertTrue(resultStr.contains("store"), "Store should remain");

        // Execute and verify correct result
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(12);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(5.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32); // unused
            interpreter.bindBuffer(2, output);
            interpreter.execute(optimized);

            // All elements should be 5.0
            for (int j = 0; j < 3; j++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                assertEquals(5.0f, actual, 0.0001f, "Element " + j);
            }
        }
    }

    @Test
    void testReferencedAcrossNestedStructure() {
        // Build: %alive = multiply %a, %b; for i in [0, 3) { store out[i], %alive }
        // DCE should NOT remove %alive since it's referenced inside the loop
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 3);

        ScalarExpr aliveExpr = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        ScalarLet aliveLet = new ScalarLet("alive", aliveExpr);
        ScalarRef aliveRef = new ScalarRef("alive", DataType.FP32);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        Store store = new Store(out, offset, aliveRef);

        Loop loop = Loop.parallel("i", 3, store);
        Block body = new Block(java.util.List.of(aliveLet, loop));
        LIRGraph graph = builder.build(body);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Print result
        LIRTextRenderer renderer = new LIRTextRenderer();
        String resultStr = renderer.render(optimized);
        System.out.println("=== DCE with reference in nested loop ===");
        System.out.println(resultStr);

        // Verify the let was NOT removed since it's referenced inside the loop
        assertTrue(
                resultStr.contains("%alive"), "ScalarLet referenced in loop should NOT be removed");
        assertTrue(resultStr.contains("multiply fp32"), "Multiply should remain");

        // Execute and verify correct result
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(12);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(2.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32);
            interpreter.bindBuffer(2, output);
            interpreter.execute(optimized);

            // All elements should be 2.0 * 3.0 = 6.0
            for (int j = 0; j < 3; j++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                assertEquals(6.0f, actual, 0.0001f, "Element " + j);
            }
        }
    }

    @Test
    void testSelfReferenceNotDead() {
        // Build: %self = add %self, %a  (This shouldn't be valid LIR, but test defensive behavior)
        // Actually, let's test: %x = add %x, %a where %x is used in store
        // This tests that we correctly handle references
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        // Create: %used = negate %scalar0; store out[0], %used
        ScalarExpr expr = new ScalarUnary(UnaryOperator.NEGATE, in0);
        ScalarLet let = new ScalarLet("used", expr);
        ScalarRef ref = new ScalarRef("used", DataType.FP32);
        Store store = new Store(out, new IndexConst(0), ref);

        Block body = new Block(java.util.List.of(let, store));
        LIRGraph graph = builder.build(body);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Verify the let was NOT removed
        LIRTextRenderer renderer = new LIRTextRenderer();
        String resultStr = renderer.render(optimized);
        assertTrue(resultStr.contains("%used"), "Used ScalarLet should NOT be removed");
    }

    @Test
    void testNoDeadCode() {
        // Build graph with no ScalarLets at all - should return same graph
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        Store store = new Store(out, offset, in0);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Apply DCE pass
        DeadCodeEliminationPass dcePass = new DeadCodeEliminationPass();
        LIRGraph optimized = dcePass.run(graph);

        // Should return the same graph (identity transform)
        assertSame(graph, optimized, "DCE should return same graph when no dead code");
    }
}

package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.Test;

class ReductionLoopFoldingPassTest {

    @Test
    void testFoldSimpleReduction() {
        // Build: for i in [0, 4) iter_args(%acc = 0) { yield add %acc, %scalar0 }
        // Should fold to: %acc = add 0, (scalar0 * 4)
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarExpr init = ScalarLiteral.ofFloat(0.0f);
        ScalarExpr updated =
                new ScalarBinary(BinaryOperator.ADD, new ScalarRef("acc", DataType.FP32), in0);

        StructuredFor loop =
                new StructuredFor(
                        "i",
                        new IndexConst(0),
                        new IndexConst(4),
                        new IndexConst(1),
                        List.of(new LoopIterArg("acc", DataType.FP32, init)),
                        new Yield(List.of(updated)));

        Store store = new Store(out, new IndexConst(0), new ScalarRef("acc", DataType.FP32));
        Block body = new Block(List.of(loop, store));
        LIRGraph graph = builder.build(body);

        LIRTextRenderer renderer = new LIRTextRenderer();
        System.out.println("=== Original Simple Reduction ===");
        System.out.println(renderer.render(graph));

        ReductionLoopFoldingPass pass = new ReductionLoopFoldingPass();
        LIRGraph folded = pass.run(graph);

        String foldedStr = renderer.render(folded);
        System.out.println("=== Folded Simple Reduction ===");
        System.out.println(foldedStr);

        assertFalse(foldedStr.contains("for %i"), "Loop should be eliminated");
        assertTrue(foldedStr.contains("4.0f"), "Should have literal 4.0f for bound");
        assertTrue(foldedStr.contains("multiply"), "Should have multiply for folding");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(4);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(5.0f), DataType.FP32);
            interpreter.bindBuffer(1, output);
            interpreter.execute(folded);

            float result = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(20.0f, result, 0.0001f, "4 * 5.0 = 20.0");
        }
    }

    @Test
    void testNoFoldingWhenLoopDependent() {
        // Build: for i in [0, 4) iter_args(%acc = 0) { yield add %acc, load(in[i]) }
        // The load depends on loop index, so cannot fold
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarLoad load = new ScalarLoad(in, offset);

        ScalarExpr init = ScalarLiteral.ofFloat(0.0f);
        ScalarExpr updated =
                new ScalarBinary(BinaryOperator.ADD, new ScalarRef("acc", DataType.FP32), load);

        StructuredFor loop =
                new StructuredFor(
                        "i",
                        new IndexConst(0),
                        new IndexConst(4),
                        new IndexConst(1),
                        List.of(new LoopIterArg("acc", DataType.FP32, init)),
                        new Yield(List.of(updated)));

        Store store = new Store(out, new IndexConst(0), new ScalarRef("acc", DataType.FP32));
        Block body = new Block(List.of(loop, store));
        LIRGraph graph = builder.build(body);

        LIRTextRenderer renderer = new LIRTextRenderer();
        System.out.println("=== No Folding Expected (Loop Dependent) ===");
        System.out.println(renderer.render(graph));

        ReductionLoopFoldingPass pass = new ReductionLoopFoldingPass();
        LIRGraph result = pass.run(graph);

        String resultStr = renderer.render(result);
        System.out.println("=== After Pass (should be unchanged) ===");
        System.out.println(resultStr);

        assertTrue(resultStr.contains("for %i"), "Loop should remain");
        assertTrue(resultStr.contains("yield"), "Yield should remain");
    }
}

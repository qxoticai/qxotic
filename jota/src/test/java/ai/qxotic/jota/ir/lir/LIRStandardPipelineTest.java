package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.List;
import org.junit.jupiter.api.Test;

class LIRStandardPipelineTest {

    @Test
    void testDefaultPipelineRuns() {
        // Build a simple graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Run standard pipeline
        LIRGraph optimized = LIRStandardPipeline.optimize(graph);

        // Verify graph is still valid
        assertNotNull(optimized);
        assertEquals(1, optimized.inputs().size());
        assertEquals(1, optimized.outputs().size());
    }

    @Test
    void testPipelineWithMultipleIterations() {
        // Build a graph with opportunities for iterative optimization
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        // Redundant computation: (i * 4) appears twice
        IndexExpr offset1 = IndexBinary.multiply(i, new IndexConst(4));
        IndexExpr offset2 = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr load = new ScalarLoad(in0, offset1);
        Store store = new Store(out, offset2, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Run with 2 iterations
        LIRGraph optimized = LIRStandardPipeline.optimize(graph, 2);

        // Verify graph is optimized
        assertNotNull(optimized);
    }

    @Test
    void testPipelineWithVerbose() {
        // Build a simple graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Run with verbose output
        LIRStandardPipeline pipeline = new LIRStandardPipeline().withVerbose(true);
        System.out.println("Pipeline configuration:");
        System.out.println(pipeline);

        LIRGraph optimized = pipeline.run(graph);

        assertNotNull(optimized);
    }

    @Test
    void testPipelineWithCustomPass() {
        // Build a simple graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(4));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Create pipeline with custom pass
        LIRPass customPass =
                new LIRPass() {
                    @Override
                    public LIRGraph run(LIRGraph graph) {
                        // Custom pass that does nothing but verifies it runs
                        return graph;
                    }

                    @Override
                    public String name() {
                        return "CustomTestPass";
                    }
                };

        LIRStandardPipeline pipeline = new LIRStandardPipeline().withPass(customPass);
        LIRGraph optimized = pipeline.run(graph);

        assertNotNull(optimized);

        // Verify custom pass is in the list
        List<LIRPass> passes = pipeline.getPasses();
        boolean foundCustom = passes.stream().anyMatch(p -> p.name().equals("CustomTestPass"));
        assertTrue(foundCustom, "Custom pass should be in pipeline");
    }

    @Test
    void testPipelineOptimizesMatmulPattern() {
        // Build a matmul-like pattern with loop invariants
        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput scalarA = builder.addScalarInput(DataType.FP32);
        ScalarInput scalarB = builder.addScalarInput(DataType.FP32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        ScalarRef aRef = new ScalarRef("scalar0", DataType.FP32);
        ScalarRef bRef = new ScalarRef("scalar1", DataType.FP32);
        ScalarExpr prod = new ScalarBinary(BinaryOperator.MULTIPLY, aRef, bRef);

        ScalarExpr init = ScalarLiteral.ofFloat(0.0f);
        ScalarExpr updated =
                new ScalarBinary(BinaryOperator.ADD, new ScalarRef("acc", DataType.FP32), prod);
        StructuredFor loop =
                new StructuredFor(
                        "i",
                        new IndexConst(0),
                        new IndexConst(3),
                        new IndexConst(1),
                        List.of(new LoopIterArg("acc", DataType.FP32, init)),
                        new Yield(List.of(updated)));

        Store store = new Store(out, new IndexConst(0), new ScalarRef("acc", DataType.FP32));
        Block body = new Block(List.of(loop, store));

        LIRGraph graph = builder.build(body);

        // Print original
        System.out.println("=== Before Pipeline ===");
        System.out.println(new LIRTextRenderer().render(graph));

        // Run pipeline
        LIRStandardPipeline pipeline = new LIRStandardPipeline().withVerbose(false);
        LIRGraph optimized = pipeline.run(graph);

        // Print optimized
        System.out.println("=== After Pipeline ===");
        System.out.println(new LIRTextRenderer().render(optimized));

        // The loop should be simplified (loop folding or hoisting)
        assertNotNull(optimized);
    }

    @Test
    void testPipelineEliminatesDeadCode() {
        // Build a graph with dead code
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

        // %dead = multiply %scalar0, %scalar1  // never used
        // %alive = add %scalar0, %scalar1
        // store %out[0], %alive

        ScalarInput in0 = builder.addScalarInput(DataType.FP32);
        ScalarInput in1 = builder.addScalarInput(DataType.FP32);

        ScalarExpr deadExpr = new ScalarBinary(BinaryOperator.MULTIPLY, in0, in1);
        ScalarExpr aliveExpr = new ScalarBinary(BinaryOperator.ADD, in0, in1);

        ScalarLet deadLet = new ScalarLet("dead", deadExpr);
        ScalarLet aliveLet = new ScalarLet("alive", aliveExpr);
        ScalarRef aliveRef = new ScalarRef("alive", DataType.FP32);
        Store store = new Store(out, new IndexConst(0), aliveRef);

        Block body = new Block(List.of(deadLet, aliveLet, store));
        LIRGraph graph = builder.build(body);

        // Print original
        System.out.println("=== Before Pipeline (with dead code) ===");
        System.out.println(new LIRTextRenderer().render(graph));

        // Run pipeline
        LIRGraph optimized = LIRStandardPipeline.optimize(graph);

        // Print optimized
        System.out.println("=== After Pipeline (dead code eliminated) ===");
        System.out.println(new LIRTextRenderer().render(optimized));

        // Verify dead code is eliminated
        String result = new LIRTextRenderer().render(optimized);
        assertFalse(result.contains("%dead"), "Dead code should be eliminated");
        assertTrue(result.contains("%alive"), "Alive code should remain");
    }

    @Test
    void testPipelineDescription() {
        LIRStandardPipeline pipeline = new LIRStandardPipeline();
        String description = pipeline.toString();

        System.out.println("Pipeline description:");
        System.out.println(description);

        // Verify description contains expected passes
        assertTrue(description.contains("Canonicalization"));
        assertTrue(description.contains("CommonSubexpressionElimination"));
        assertTrue(description.contains("LoopInvariantHoisting"));
        assertTrue(description.contains("ReductionLoopFolding"));
        assertTrue(description.contains("IndexSimplification"));
        assertTrue(description.contains("DeadCodeElimination"));
    }

    @Test
    void testGetPassesReturnsCopy() {
        LIRStandardPipeline pipeline = new LIRStandardPipeline();
        List<LIRPass> passes1 = pipeline.getPasses();
        List<LIRPass> passes2 = pipeline.getPasses();

        // Should return different copies
        assertNotSame(passes1, passes2);

        // Modifying one should not affect the other
        passes1.clear();
        assertFalse(pipeline.getPasses().isEmpty());
    }

    @Test
    void testInvalidIterations() {
        LIRStandardPipeline pipeline = new LIRStandardPipeline();

        assertThrows(IllegalArgumentException.class, () -> pipeline.withIterations(0));
        assertThrows(IllegalArgumentException.class, () -> pipeline.withIterations(-1));
    }
}

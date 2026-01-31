package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class LIRPassManagerTest {

    @Test
    void testPassManagerChaining() {
        LIRPassManager manager = new LIRPassManager();

        assertEquals(0, manager.size());
        assertTrue(manager.isEmpty());

        manager.add(new IndexSimplificationPass()).add(new CommonSubexpressionElimination());

        assertEquals(2, manager.size());
        assertFalse(manager.isEmpty());
    }

    @Test
    void testIndexSimplificationPass() {
        // Build a graph with simplifiable index expressions: i * 1 + 0
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        // Create: (i * 1 + 0) * 4 - should simplify to i * 4
        IndexExpr timesOne = new IndexBinary(IndexBinary.IndexBinaryOp.MULTIPLY, i, IndexConst.ONE);
        IndexExpr plusZero =
                new IndexBinary(IndexBinary.IndexBinaryOp.ADD, timesOne, IndexConst.ZERO);
        IndexExpr offset =
                new IndexBinary(
                        IndexBinary.IndexBinaryOp.MULTIPLY, plusZero, new IndexConst(Float.BYTES));

        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Apply simplification
        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        // The graph should be functionally equivalent
        assertNotNull(simplified);

        // Verify semantics are preserved
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(4 * Float.BYTES);
            MemorySegment output = arena.allocate(4 * Float.BYTES);

            for (int idx = 0; idx < 4; idx++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx * 10.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(simplified);

            for (int idx = 0; idx < 4; idx++) {
                assertEquals(idx * 10.0f, output.getAtIndex(ValueLayout.JAVA_FLOAT, idx), 1e-6f);
            }
        }
    }

    @Test
    void testCommonSubexpressionElimination() {
        // Build a graph with duplicate expressions
        // output[i] = (input[i * 4] + input[i * 4]) where i * 4 appears twice
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");

        // Create two independent but equal offset expressions
        IndexExpr offset1 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        IndexExpr offset2 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        // These are structurally equal but different instances
        assertNotSame(offset1, offset2);
        assertEquals(offset1, offset2);

        ScalarExpr load1 = new ScalarLoad(in0, offset1);
        ScalarExpr load2 = new ScalarLoad(in0, offset2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, offset1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Apply CSE
        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        // Extract and verify CSE worked - the offset expressions should now be same instance
        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();
        ScalarLoad optLoad1 = (ScalarLoad) optSum.left();
        ScalarLoad optLoad2 = (ScalarLoad) optSum.right();

        // After CSE, the offsets should be the same instance
        assertSame(optLoad1.offset(), optLoad2.offset());
        // And the loads themselves should be the same instance
        assertSame(optLoad1, optLoad2);

        // Verify semantics are preserved
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(4 * Float.BYTES);
            MemorySegment output = arena.allocate(4 * Float.BYTES);

            for (int idx = 0; idx < 4; idx++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(optimized);

            for (int idx = 0; idx < 4; idx++) {
                float expected = 2 * (idx + 1.0f); // input[i] + input[i]
                assertEquals(expected, output.getAtIndex(ValueLayout.JAVA_FLOAT, idx), 1e-6f);
            }
        }
    }

    @Test
    void testPassPipeline() {
        // Test running multiple passes in sequence
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");

        // Create redundant expressions: ((i * 1) * 4) and ((i * 1) * 4) again
        IndexExpr timesOne1 =
                new IndexBinary(IndexBinary.IndexBinaryOp.MULTIPLY, i, IndexConst.ONE);
        IndexExpr offset1 =
                new IndexBinary(
                        IndexBinary.IndexBinaryOp.MULTIPLY, timesOne1, new IndexConst(Float.BYTES));

        IndexExpr timesOne2 =
                new IndexBinary(IndexBinary.IndexBinaryOp.MULTIPLY, i, IndexConst.ONE);
        IndexExpr offset2 =
                new IndexBinary(
                        IndexBinary.IndexBinaryOp.MULTIPLY, timesOne2, new IndexConst(Float.BYTES));

        ScalarExpr load1 = new ScalarLoad(in0, offset1);
        ScalarExpr load2 = new ScalarLoad(in0, offset2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, offset1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Run both passes
        LIRGraph optimized =
                new LIRPassManager()
                        .add(new IndexSimplificationPass())
                        .add(new CommonSubexpressionElimination())
                        .run(graph);

        // Verify semantics preserved
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(4 * Float.BYTES);
            MemorySegment output = arena.allocate(4 * Float.BYTES);

            for (int idx = 0; idx < 4; idx++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(optimized);

            for (int idx = 0; idx < 4; idx++) {
                float expected = 2 * (idx + 1.0f);
                assertEquals(expected, output.getAtIndex(ValueLayout.JAVA_FLOAT, idx), 1e-6f);
            }
        }
    }

    @Test
    void testIdentityPass() {
        // Test that an empty pass manager returns the same graph
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Empty pipeline
        LIRGraph result = new LIRPassManager().run(graph);

        assertSame(graph, result);
    }

    @Test
    void testCustomPass() {
        // Test custom pass that counts nodes
        int[] nodeCount = {0};
        LIRPass countingPass =
                new LIRPass() {
                    @Override
                    public LIRGraph run(LIRGraph graph) {
                        new LIRRewriter() {
                            @Override
                            public LIRNode visitLoop(Loop node) {
                                nodeCount[0]++;
                                return super.visitLoop(node);
                            }

                            @Override
                            public LIRNode visitStore(Store node) {
                                nodeCount[0]++;
                                return super.visitStore(node);
                            }
                        }.rewrite(graph);
                        return graph; // Return unchanged
                    }

                    @Override
                    public String name() {
                        return "CountingPass";
                    }
                };

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        new LIRPassManager().add(countingPass).run(graph);

        assertEquals(2, nodeCount[0]); // 1 Loop + 1 Store
    }

    @Test
    void testLIRRewriterStructuralSharing() {
        // Test that unchanged nodes return the same instance
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Identity rewriter that doesn't change anything
        LIRRewriter rewriter = new LIRRewriter();
        LIRGraph result = rewriter.rewrite(graph);

        // Should return the same graph
        assertSame(graph, result);
    }

    @Test
    void testConstantFoldingThroughSimplify() {
        // Test that constant expressions are fully evaluated
        IndexExpr expr =
                IndexBinary.add(
                        IndexBinary.multiply(new IndexConst(3), new IndexConst(4)),
                        new IndexConst(2));

        IndexExpr simplified = expr.simplify();

        // Should fold to constant 14
        assertTrue(simplified.isConstant());
        assertEquals(14, simplified.constantValue());
    }

    @Test
    void testStrengthReduction() {
        // Test that multiplication by power of 2 becomes shift
        IndexVar i = new IndexVar("i");
        IndexExpr mult = IndexBinary.multiply(i, new IndexConst(8));

        IndexExpr simplified = mult.simplify();

        // Should become shift left
        assertInstanceOf(IndexBinary.class, simplified);
        IndexBinary binary = (IndexBinary) simplified;
        assertEquals(IndexBinary.IndexBinaryOp.SHIFT_LEFT, binary.op());
        assertEquals(3, ((IndexConst) binary.right()).value()); // 8 = 2^3
    }

    @Test
    void testPassName() {
        LIRPass indexPass = new IndexSimplificationPass();
        LIRPass csePass = new CommonSubexpressionElimination();
        LIRPass canonPass = new CanonicalizationPass();

        assertEquals("IndexSimplification", indexPass.name());
        assertEquals("CommonSubexpressionElimination", csePass.name());
        assertEquals("Canonicalization", canonPass.name());
    }

    @Test
    void testCanonicalizationPassIndexCommutative() {
        // Test that i + 4 and 4 + i become the same canonical form (4 + i, constant first)
        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        // i + 4 (variable first)
        IndexExpr expr1 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, four);
        // 4 + i (constant first - already canonical)
        IndexExpr expr2 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, four, i);

        // Build two graphs with these expressions
        LIRGraph.Builder builder1 = LIRGraph.builder();
        BufferRef in1 = builder1.addContiguousInput(DataType.FP32, 8);
        BufferRef out1 = builder1.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load1 = new ScalarLoad(in1, expr1);
        Store store1 = new Store(out1, expr1, load1);
        Loop loop1 = Loop.parallel("i", 8, store1);
        LIRGraph graph1 = builder1.build(loop1);

        LIRGraph.Builder builder2 = LIRGraph.builder();
        BufferRef in2 = builder2.addContiguousInput(DataType.FP32, 8);
        BufferRef out2 = builder2.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load2 = new ScalarLoad(in2, expr2);
        Store store2 = new Store(out2, expr2, load2);
        Loop loop2 = Loop.parallel("i", 8, store2);
        LIRGraph graph2 = builder2.build(loop2);

        // Canonicalize both
        CanonicalizationPass pass = new CanonicalizationPass();
        LIRGraph canon1 = pass.run(graph1);
        LIRGraph canon2 = pass.run(graph2);

        // Extract the canonicalized index expressions
        Loop canonLoop1 = (Loop) canon1.body();
        Store canonStore1 = (Store) canonLoop1.body();
        IndexBinary canonIdx1 = (IndexBinary) canonStore1.offset();

        Loop canonLoop2 = (Loop) canon2.body();
        Store canonStore2 = (Store) canonLoop2.body();
        IndexBinary canonIdx2 = (IndexBinary) canonStore2.offset();

        // Both should now have constant on the left (canonical form)
        assertInstanceOf(IndexConst.class, canonIdx1.left());
        assertInstanceOf(IndexVar.class, canonIdx1.right());
        assertInstanceOf(IndexConst.class, canonIdx2.left());
        assertInstanceOf(IndexVar.class, canonIdx2.right());

        // The expressions should be structurally equal
        assertEquals(canonIdx1, canonIdx2);
    }

    @Test
    void testCanonicalizationPassScalarCommutative() {
        // Test that load(a) + load(b) and load(b) + load(a) become same form
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        // load(in1) + load(in0) - in1 has higher buffer id, so should be reordered
        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn1, loadIn0); // in1 first
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Canonicalize
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        // Extract the sum
        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        // After canonicalization, in0 (lower buffer id) should come first
        ScalarLoad leftLoad = (ScalarLoad) canonSum.left();
        ScalarLoad rightLoad = (ScalarLoad) canonSum.right();

        assertEquals(in0.id(), leftLoad.buffer().id());
        assertEquals(in1.id(), rightLoad.buffer().id());

        // Verify semantics preserved
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input0 = arena.allocate(4 * Float.BYTES);
            MemorySegment input1 = arena.allocate(4 * Float.BYTES);
            MemorySegment output = arena.allocate(4 * Float.BYTES);

            for (int idx = 0; idx < 4; idx++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx * 2.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input0);
            interpreter.bindBuffer(in1.id(), input1);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(canonicalized);

            for (int idx = 0; idx < 4; idx++) {
                float expected = (idx + 1.0f) + (idx * 2.0f);
                assertEquals(expected, output.getAtIndex(ValueLayout.JAVA_FLOAT, idx), 1e-6f);
            }
        }
    }

    @Test
    void testCanonicalizationWithCSE() {
        // Test that canonicalization enables CSE for commutative ops
        // Create: (i + 4) and (4 + i) which are different but equivalent
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        // Create i + 4 and 4 + i (different order)
        IndexExpr expr1 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, four);
        IndexExpr expr2 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, four, i);

        // Without canonicalization, these are different
        assertNotEquals(expr1, expr2);

        ScalarExpr load1 = new ScalarLoad(in0, expr1);
        ScalarExpr load2 = new ScalarLoad(in0, expr2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, expr1, sum);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        // Run canonicalization then CSE
        LIRGraph optimized =
                new LIRPassManager()
                        .add(new CanonicalizationPass())
                        .add(new CommonSubexpressionElimination())
                        .run(graph);

        // After canonicalization + CSE, the two loads should be the same instance
        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        // Both loads should now be the same instance
        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCanonicalizationNonCommutative() {
        // Test that non-commutative ops are not reordered
        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        // i - 4 (subtract is not commutative)
        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.SUBTRACT, i, four);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        // Canonicalize
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        // The expression should be unchanged (i - 4, not 4 - i)
        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexVar.class, canonIdx.left());
        assertInstanceOf(IndexConst.class, canonIdx.right());
    }

    @Test
    void testCanonicalizationNestedExpressions() {
        // Test that nested commutative expressions are all canonicalized
        // (i + j) + 4 where inner should become canonical too
        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexConst four = new IndexConst(4);

        // j + i (should become i + j since i < j alphabetically)
        IndexExpr inner = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, j, i);
        // (j + i) + 4 (should become 4 + (i + j))
        IndexExpr outer = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, inner, four);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 64);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 64);
        ScalarExpr load = new ScalarLoad(in0, outer);
        Store store = new Store(out, outer, load);
        Loop loopJ = Loop.parallel("j", 8, store);
        Loop loopI = Loop.parallel("i", 8, loopJ);

        LIRGraph graph = builder.build(loopI);

        // Canonicalize
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        // Extract the outer expression
        Loop canonLoopI = (Loop) canonicalized.body();
        Loop canonLoopJ = (Loop) canonLoopI.body();
        Store canonStore = (Store) canonLoopJ.body();
        IndexBinary canonOuter = (IndexBinary) canonStore.offset();

        // Outer: constant should be on left
        assertInstanceOf(IndexConst.class, canonOuter.left());
        assertInstanceOf(IndexBinary.class, canonOuter.right());

        // Inner: i should be on left (alphabetically before j)
        IndexBinary canonInner = (IndexBinary) canonOuter.right();
        assertInstanceOf(IndexVar.class, canonInner.left());
        assertInstanceOf(IndexVar.class, canonInner.right());
        assertEquals("i", ((IndexVar) canonInner.left()).name());
        assertEquals("j", ((IndexVar) canonInner.right()).name());
    }
}

package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class CommonSubexpressionEliminationTest {

    @Test
    void testCSEIdenticalIndexBinary() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");

        IndexExpr offset1 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        IndexExpr offset2 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr load1 = new ScalarLoad(in0, offset1);
        ScalarExpr load2 = new ScalarLoad(in0, offset2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, offset1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        assertNotSame(offset1, offset2);
        assertEquals(offset1, offset2);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        ScalarLoad optLoad1 = (ScalarLoad) optSum.left();
        ScalarLoad optLoad2 = (ScalarLoad) optSum.right();

        assertSame(optLoad1.offset(), optLoad2.offset());
    }

    @Test
    void testCSEIdenticalIndexConstants() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");

        IndexExpr offsetI = IndexBinary.multiply(i, new IndexConst(4));
        IndexExpr offsetJ = IndexBinary.multiply(j, new IndexConst(4));

        ScalarExpr loadI = new ScalarLoad(in0, offsetI);
        ScalarExpr loadJ = new ScalarLoad(in0, offsetJ);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadI, loadJ);
        Store store = new Store(out, offsetI, sum);
        Loop loopJ = Loop.parallel("j", 2, store);
        Loop loopI = Loop.parallel("i", 4, loopJ);

        LIRGraph graph = builder.build(loopI);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoopI = (Loop) optimized.body();
        Loop optLoopJ = (Loop) optLoopI.body();
        Store optStore = (Store) optLoopJ.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        ScalarLoad optLoadI = (ScalarLoad) optSum.left();
        ScalarLoad optLoadJ = (ScalarLoad) optSum.right();

        IndexBinary optOffsetI = (IndexBinary) optLoadI.offset();
        IndexBinary optOffsetJ = (IndexBinary) optLoadJ.offset();

        assertSame(optOffsetI.right(), optOffsetJ.right());
    }

    @Test
    void testCSEIndexAfterSimplification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr timesOne1 = IndexBinary.multiply(i, IndexConst.ONE);
        IndexExpr timesOne2 = IndexBinary.multiply(i, IndexConst.ONE);

        ScalarExpr load1 = new ScalarLoad(in0, timesOne1);
        ScalarExpr load2 = new ScalarLoad(in0, timesOne2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, timesOne1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph simplified = new IndexSimplificationPass().run(graph);
        LIRGraph withCSE = new CommonSubexpressionElimination().run(simplified);

        Loop cseLoop = (Loop) withCSE.body();
        Store cseStore = (Store) cseLoop.body();
        ScalarBinary cseSum = (ScalarBinary) cseStore.value();

        assertSame(cseSum.left(), cseSum.right());
    }

    @Test
    void testCSEIdenticalScalarBinary() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);

        ScalarExpr sum1 = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        ScalarExpr sum2 = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        ScalarExpr total = new ScalarBinary(BinaryOperator.ADD, sum1, sum2);
        Store store = new Store(out, offset, total);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        assertNotSame(sum1, sum2);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optTotal = (ScalarBinary) optStore.value();

        assertSame(optTotal.left(), optTotal.right());
    }

    @Test
    void testCSEIdenticalScalarLoad() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr load1 = new ScalarLoad(in0, offset);
        ScalarExpr load2 = new ScalarLoad(in0, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEIdenticalScalarLiteral() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarLiteral lit = new ScalarLiteral(Float.floatToRawIntBits(3.0f), DataType.FP32);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, lit, lit);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEMultipleCommonSubexprs() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);

        ScalarExpr sum1 = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        ScalarExpr sum2 = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        ScalarExpr sum3 = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        ScalarExpr total =
                new ScalarBinary(
                        BinaryOperator.ADD, sum1, new ScalarBinary(BinaryOperator.ADD, sum2, sum3));
        Store store = new Store(out, offset, total);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        assertNotNull(optimized);
        assertNotSame(graph, optimized);
    }

    @Test
    void testCSENoDuplicates() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset1 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        IndexExpr offset2 = IndexBinary.multiply(i, new IndexConst(Float.BYTES * 2));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset1);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        Store store = new Store(out, offset1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertNotSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEAlreadyOptimized() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized = new CommonSubexpressionElimination().run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarExpr optValue = optStore.value();

        Loop origLoop = (Loop) graph.body();
        Store origStore = (Store) origLoop.body();
        ScalarExpr origValue = origStore.value();

        assertEquals(origValue, optValue);
    }

    @Test
    void testCSEAfterCanonicalization() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexExpr expr1 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, new IndexConst(4));
        IndexExpr expr2 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, new IndexConst(4), i);

        ScalarExpr load1 = new ScalarLoad(in0, expr1);
        ScalarExpr load2 = new ScalarLoad(in0, expr2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, expr1, sum);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        assertNotEquals(expr1, expr2);

        LIRGraph optimized =
                new LIRPassManager()
                        .add(new CanonicalizationPass())
                        .add(new CommonSubexpressionElimination())
                        .run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEAfterSimplification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset1 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        IndexExpr offset2 = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr load1 = new ScalarLoad(in0, offset1);
        ScalarExpr load2 = new ScalarLoad(in0, offset2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, offset1, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized =
                new LIRPassManager()
                        .add(new IndexSimplificationPass())
                        .add(new CommonSubexpressionElimination())
                        .run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEPipeline() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexExpr expr1 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, new IndexConst(4));
        IndexExpr expr2 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, new IndexConst(4), i);

        ScalarExpr load1 = new ScalarLoad(in0, expr1);
        ScalarExpr load2 = new ScalarLoad(in0, expr2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, expr1, sum);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        LIRGraph optimized =
                new LIRPassManager()
                        .add(new CanonicalizationPass())
                        .add(new IndexSimplificationPass())
                        .add(new CommonSubexpressionElimination())
                        .run(graph);

        Loop optLoop = (Loop) optimized.body();
        Store optStore = (Store) optLoop.body();
        ScalarBinary optSum = (ScalarBinary) optStore.value();

        assertSame(optSum.left(), optSum.right());
    }

    @Test
    void testCSEPreservesSemantics() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn0, loadIn1);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input0 = arena.allocate(4 * Float.BYTES);
            MemorySegment input1 = arena.allocate(4 * Float.BYTES);
            MemorySegment outputOriginal = arena.allocate(4 * Float.BYTES);
            MemorySegment outputCSE = arena.allocate(4 * Float.BYTES);

            for (int idx = 0; idx < 4; idx++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx * 2.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input0);
            interpreter.bindBuffer(in1.id(), input1);
            interpreter.bindBuffer(out.id(), outputOriginal);
            interpreter.execute(graph);

            LIRGraph withCSE = new CommonSubexpressionElimination().run(graph);
            interpreter.bindBuffer(out.id(), outputCSE);
            interpreter.execute(withCSE);

            for (int idx = 0; idx < 4; idx++) {
                assertEquals(
                        outputOriginal.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        outputCSE.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        1e-6f);
            }
        }
    }
}

package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class CanonicalizationPassTest {

    @Test
    void testCanonicalizeIndexCommutativeAdd() {
        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, four);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexConst.class, canonIdx.left());
        assertInstanceOf(IndexVar.class, canonIdx.right());
        assertEquals(4, ((IndexConst) canonIdx.left()).value());
        assertEquals("i", ((IndexVar) canonIdx.right()).name());
    }

    @Test
    void testCanonicalizeIndexCommutativeMultiply() {
        IndexVar i = new IndexVar("i");
        IndexConst eight = new IndexConst(8);

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.MULTIPLY, i, eight);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 16);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 16);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 16, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexConst.class, canonIdx.left());
        assertInstanceOf(IndexVar.class, canonIdx.right());
        assertEquals(8, ((IndexConst) canonIdx.left()).value());
        assertEquals("i", ((IndexVar) canonIdx.right()).name());
    }

    @Test
    void testCanonicalizeIndexNonCommutativeSubtract() {
        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.SUBTRACT, i, four);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexVar.class, canonIdx.left());
        assertInstanceOf(IndexConst.class, canonIdx.right());
        assertEquals("i", ((IndexVar) canonIdx.left()).name());
        assertEquals(4, ((IndexConst) canonIdx.right()).value());
    }

    @Test
    void testCanonicalizeIndexNonCommutativeDivide() {
        IndexVar i = new IndexVar("i");
        IndexConst eight = new IndexConst(8);

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.DIVIDE, i, eight);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexVar.class, canonIdx.left());
        assertInstanceOf(IndexConst.class, canonIdx.right());
        assertEquals("i", ((IndexVar) canonIdx.left()).name());
        assertEquals(8, ((IndexConst) canonIdx.right()).value());
    }

    @Test
    void testCanonicalizeIndexConstantsFirst() {
        IndexConst four = new IndexConst(4);
        IndexVar i = new IndexVar("i");

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, four, i);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexConst.class, canonIdx.left());
        assertInstanceOf(IndexVar.class, canonIdx.right());
    }

    @Test
    void testCanonicalizeIndexVarAlphabetical() {
        IndexVar j = new IndexVar("j");
        IndexVar i = new IndexVar("i");

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, j, i);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 64);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 64);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loopJ = Loop.parallel("j", 8, store);
        Loop loopI = Loop.parallel("i", 8, loopJ);
        LIRGraph graph = builder.build(loopI);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoopI = (Loop) canonicalized.body();
        Loop canonLoopJ = (Loop) canonLoopI.body();
        Store canonStore = (Store) canonLoopJ.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexVar.class, canonIdx.left());
        assertInstanceOf(IndexVar.class, canonIdx.right());
        assertEquals("i", ((IndexVar) canonIdx.left()).name());
        assertEquals("j", ((IndexVar) canonIdx.right()).name());
    }

    @Test
    void testCanonicalizeScalarCommutativeAdd() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn1, loadIn0);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        ScalarLoad leftLoad = (ScalarLoad) canonSum.left();
        ScalarLoad rightLoad = (ScalarLoad) canonSum.right();

        assertEquals(in0.id(), leftLoad.buffer().id());
        assertEquals(in1.id(), rightLoad.buffer().id());

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
    void testCanonicalizeScalarCommutativeMultiply() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr product = new ScalarBinary(BinaryOperator.MULTIPLY, loadIn1, loadIn0);
        Store store = new Store(out, offset, product);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonProduct = (ScalarBinary) canonStore.value();

        ScalarLoad leftLoad = (ScalarLoad) canonProduct.left();
        ScalarLoad rightLoad = (ScalarLoad) canonProduct.right();

        assertEquals(in0.id(), leftLoad.buffer().id());
        assertEquals(in1.id(), rightLoad.buffer().id());
    }

    @Test
    void testCanonicalizeScalarNonCommutativeSubtract() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr diff = new ScalarBinary(BinaryOperator.SUBTRACT, loadIn1, loadIn0);
        Store store = new Store(out, offset, diff);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonDiff = (ScalarBinary) canonStore.value();

        assertSame(in1.id(), ((ScalarLoad) canonDiff.left()).buffer().id());
        assertSame(in0.id(), ((ScalarLoad) canonDiff.right()).buffer().id());
    }

    @Test
    void testCanonicalizeScalarLiteralTypeOrdering() {
        ScalarLiteral fp32Lit = new ScalarLiteral(Float.floatToRawIntBits(3.0f), DataType.FP32);
        ScalarLiteral i32Lit = new ScalarLiteral(5, DataType.I32);

        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, i32Lit, fp32Lit);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);
        Store store = new Store(out, new IndexConst(0), sum);
        LIRGraph graph = builder.build(store);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Store canonStore = (Store) canonicalized.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        assertInstanceOf(ScalarLiteral.class, canonSum.left());
        assertEquals(DataType.FP32, ((ScalarLiteral) canonSum.left()).dataType());
        assertInstanceOf(ScalarLiteral.class, canonSum.right());
        assertEquals(DataType.I32, ((ScalarLiteral) canonSum.right()).dataType());
    }

    @Test
    void testCanonicalizeScalarBufferIdOrdering() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn1, loadIn0);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        ScalarLoad leftLoad = (ScalarLoad) canonSum.left();
        ScalarLoad rightLoad = (ScalarLoad) canonSum.right();

        assertEquals(in1.id(), leftLoad.buffer().id());
        assertEquals(in0.id(), rightLoad.buffer().id());
    }

    @Test
    void testCanonicalizeNestedIndexExpressions() {
        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexConst four = new IndexConst(4);

        IndexExpr inner = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, j, i);
        IndexExpr outer = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, inner, four);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 64);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 64);
        ScalarExpr load = new ScalarLoad(in0, outer);
        Store store = new Store(out, outer, load);
        Loop loopJ = Loop.parallel("j", 8, store);
        Loop loopI = Loop.parallel("i", 8, loopJ);
        LIRGraph graph = builder.build(loopI);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoopI = (Loop) canonicalized.body();
        Loop canonLoopJ = (Loop) canonLoopI.body();
        Store canonStore = (Store) canonLoopJ.body();
        IndexBinary canonOuter = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexConst.class, canonOuter.left());
        assertInstanceOf(IndexBinary.class, canonOuter.right());

        IndexBinary canonInner = (IndexBinary) canonOuter.right();
        assertInstanceOf(IndexVar.class, canonInner.left());
        assertInstanceOf(IndexVar.class, canonInner.right());
        assertEquals("i", ((IndexVar) canonInner.left()).name());
        assertEquals("j", ((IndexVar) canonInner.right()).name());
    }

    @Test
    void testCanonicalizeAlreadyCanonical() {
        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        IndexExpr expr = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, four, i);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 8, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        IndexBinary canonIdx = (IndexBinary) canonStore.offset();

        assertInstanceOf(IndexConst.class, canonIdx.left());
        assertInstanceOf(IndexVar.class, canonIdx.right());
        assertEquals(4, ((IndexConst) canonIdx.left()).value());
        assertEquals("i", ((IndexVar) canonIdx.right()).name());
    }

    @Test
    void testCanonicalizeWithLoads() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef in1 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));

        ScalarExpr loadIn0 = new ScalarLoad(in0, offset);
        ScalarExpr loadIn1 = new ScalarLoad(in1, offset);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, loadIn1, loadIn0);
        Store store = new Store(out, offset, sum);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);
        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Loop canonLoop = (Loop) canonicalized.body();
        Store canonStore = (Store) canonLoop.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        ScalarLoad leftLoad = (ScalarLoad) canonSum.left();
        ScalarLoad rightLoad = (ScalarLoad) canonSum.right();

        assertTrue(in0.id() < in1.id());
        assertEquals(in0.id(), leftLoad.buffer().id());
        assertEquals(in1.id(), rightLoad.buffer().id());
    }

    @Test
    void testCanonicalizeWithCasts() {
        ScalarInput scalarIn = new ScalarInput(0, DataType.I32);
        ScalarExpr cast = new ScalarCast(scalarIn, DataType.FP32);

        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, cast, cast);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);
        Store store = new Store(out, new IndexConst(0), sum);
        LIRGraph graph = builder.build(store);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);

        Store canonStore = (Store) canonicalized.body();
        ScalarBinary canonSum = (ScalarBinary) canonStore.value();

        assertInstanceOf(ScalarCast.class, canonSum.left());
        assertInstanceOf(ScalarCast.class, canonSum.right());
    }

    @Test
    void testCanonicalizationEnablesCSE() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexConst four = new IndexConst(4);

        IndexExpr expr1 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, i, four);
        IndexExpr expr2 = new IndexBinary(IndexBinary.IndexBinaryOp.ADD, four, i);

        ScalarExpr load1 = new ScalarLoad(in0, expr1);
        ScalarExpr load2 = new ScalarLoad(in0, expr2);
        ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, load1, load2);
        Store store = new Store(out, expr1, sum);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        assertNotEquals(expr1, expr2);

        LIRGraph canonicalized = new CanonicalizationPass().run(graph);
        LIRGraph withCSE = new CommonSubexpressionElimination().run(canonicalized);

        Loop cseLoop = (Loop) withCSE.body();
        Store cseStore = (Store) cseLoop.body();
        ScalarBinary cseSum = (ScalarBinary) cseStore.value();

        assertSame(cseSum.left(), cseSum.right());
    }

    @Test
    void testCanonicalizationPreservesSemantics() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 8);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 8);

        IndexVar i = new IndexVar("i");
        IndexExpr offset = IndexBinary.multiply(i, new IndexConst(Float.BYTES));
        ScalarExpr load = new ScalarLoad(in0, offset);
        Store store = new Store(out, offset, load);
        Loop loop = Loop.parallel("i", 8, store);

        LIRGraph graph = builder.build(loop);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(8 * Float.BYTES);
            MemorySegment outputOriginal = arena.allocate(8 * Float.BYTES);
            MemorySegment outputCanonical = arena.allocate(8 * Float.BYTES);

            for (int idx = 0; idx < 8; idx++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), outputOriginal);
            interpreter.execute(graph);

            LIRGraph canonicalized = new CanonicalizationPass().run(graph);
            interpreter.bindBuffer(out.id(), outputCanonical);
            interpreter.execute(canonicalized);

            for (int idx = 0; idx < 8; idx++) {
                assertEquals(
                        outputOriginal.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        outputCanonical.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        1e-6f);
            }
        }
    }
}

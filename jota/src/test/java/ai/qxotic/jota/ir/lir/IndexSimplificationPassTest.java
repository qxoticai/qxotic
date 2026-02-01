package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class IndexSimplificationPassTest {

    @Test
    void testSimplifyAddZero() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.add(i, IndexConst.ZERO);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        assertInstanceOf(IndexVar.class, simpStore.offset());
        assertEquals("i", ((IndexVar) simpStore.offset()).name());
    }

    @Test
    void testSimplifyMultiplyOne() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.multiply(i, IndexConst.ONE);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        assertInstanceOf(IndexVar.class, simpStore.offset());
        assertEquals("i", ((IndexVar) simpStore.offset()).name());
    }

    @Test
    void testSimplifyDivideOne() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.divide(i, IndexConst.ONE);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        assertInstanceOf(IndexVar.class, simpStore.offset());
        assertEquals("i", ((IndexVar) simpStore.offset()).name());
    }

    @Test
    void testSimplifyMultiplyByOneInNested() {
        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");
        IndexExpr timesOne = IndexBinary.multiply(j, IndexConst.ONE);
        IndexExpr expr = IndexBinary.add(i, timesOne);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 16);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 16);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loopJ = Loop.parallel("j", 4, store);
        Loop loopI = Loop.parallel("i", 4, loopJ);
        LIRGraph graph = builder.build(loopI);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoopI = (Loop) simplified.body();
        Loop simpLoopJ = (Loop) simpLoopI.body();
        Store simpStore = (Store) simpLoopJ.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertInstanceOf(IndexVar.class, simpExpr.right());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertEquals("j", ((IndexVar) simpExpr.right()).name());
    }

    @Test
    void testSimplifyMultiplyByZero() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.multiply(i, IndexConst.ZERO);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        assertInstanceOf(IndexConst.class, simpStore.offset());
        assertEquals(0, ((IndexConst) simpStore.offset()).value());
    }

    @Test
    void testSimplifyConstantAdd() {
        IndexExpr expr =
                IndexBinary.add(
                        IndexBinary.multiply(new IndexConst(3), new IndexConst(4)),
                        new IndexConst(2));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        IndexExpr simplifiedExpr = simpStore.offset();
        assertTrue(simplifiedExpr.isConstant());
        assertEquals(14, simplifiedExpr.constantValue());
    }

    @Test
    void testSimplifyConstantMultiply() {
        IndexExpr expr =
                IndexBinary.multiply(
                        IndexBinary.add(new IndexConst(2), new IndexConst(3)), new IndexConst(5));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        IndexExpr simplifiedExpr = simpStore.offset();
        assertTrue(simplifiedExpr.isConstant());
        assertEquals(25, simplifiedExpr.constantValue());
    }

    @Test
    void testSimplifyMultiplyByPowerOf2() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.multiply(i, new IndexConst(8));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 16);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 16);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 16, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.SHIFT_LEFT, simpExpr.op());
        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertInstanceOf(IndexConst.class, simpExpr.right());
        assertEquals(3, ((IndexConst) simpExpr.right()).value());
    }

    @Test
    void testSimplifyMultiplyBy16() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.multiply(i, new IndexConst(16));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 32);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 32);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 32, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.SHIFT_LEFT, simpExpr.op());
        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertInstanceOf(IndexConst.class, simpExpr.right());
        assertEquals(4, ((IndexConst) simpExpr.right()).value());
    }

    @Test
    void testSimplifyModuloByPowerOf2() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.modulo(i, new IndexConst(8));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 16);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 16);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 16, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.BITWISE_AND, simpExpr.op());
        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertInstanceOf(IndexConst.class, simpExpr.right());
        assertEquals(7, ((IndexConst) simpExpr.right()).value());
    }

    @Test
    void testSimplifyModuloByNonPowerOf2() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.modulo(i, new IndexConst(7));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 16);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 16);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 16, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.MODULO, simpExpr.op());
    }

    @Test
    void testSimplifyDeepNesting() {
        IndexVar i = new IndexVar("i");
        IndexExpr timesOne1 = IndexBinary.multiply(i, IndexConst.ONE);
        IndexExpr timesOne2 = IndexBinary.multiply(timesOne1, IndexConst.ONE);
        IndexExpr timesOne3 = IndexBinary.multiply(timesOne2, IndexConst.ONE);

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, timesOne3);
        Store store = new Store(out, timesOne3, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();

        assertInstanceOf(IndexVar.class, simpStore.offset());
        assertEquals("i", ((IndexVar) simpStore.offset()).name());
    }

    @Test
    void testSimplifyAlreadySimple() {
        IndexVar i = new IndexVar("i");
        IndexExpr expr = IndexBinary.multiply(i, new IndexConst(5));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.MULTIPLY, simpExpr.op());
        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertInstanceOf(IndexConst.class, simpExpr.right());
        assertEquals(5, ((IndexConst) simpExpr.right()).value());
    }

    @Test
    void testSimplifyWithComplexIndexing() {
        IndexVar i = new IndexVar("i");
        IndexExpr timesOne = IndexBinary.multiply(i, IndexConst.ONE);
        IndexExpr plusZero = IndexBinary.add(timesOne, IndexConst.ZERO);
        IndexExpr expr = IndexBinary.multiply(plusZero, new IndexConst(5));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in0 = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        ScalarExpr load = new ScalarLoad(in0, expr);
        Store store = new Store(out, expr, load);
        Loop loop = Loop.parallel("i", 4, store);
        LIRGraph graph = builder.build(loop);

        LIRGraph simplified = new IndexSimplificationPass().run(graph);

        Loop simpLoop = (Loop) simplified.body();
        Store simpStore = (Store) simpLoop.body();
        IndexBinary simpExpr = (IndexBinary) simpStore.offset();

        assertEquals(IndexBinary.IndexBinaryOp.MULTIPLY, simpExpr.op());
        assertInstanceOf(IndexVar.class, simpExpr.left());
        assertEquals("i", ((IndexVar) simpExpr.left()).name());
        assertInstanceOf(IndexConst.class, simpExpr.right());
        assertEquals(5, ((IndexConst) simpExpr.right()).value());
    }

    @Test
    void testSimplificationPreservesSemantics() {
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
            MemorySegment outputSimplified = arena.allocate(8 * Float.BYTES);

            for (int idx = 0; idx < 8; idx++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, idx, idx + 1.0f);
            }

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), outputOriginal);
            interpreter.execute(graph);

            LIRGraph simplified = new IndexSimplificationPass().run(graph);
            interpreter.bindBuffer(out.id(), outputSimplified);
            interpreter.execute(simplified);

            for (int idx = 0; idx < 8; idx++) {
                assertEquals(
                        outputOriginal.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        outputSimplified.getAtIndex(ValueLayout.JAVA_FLOAT, idx),
                        1e-6f);
            }
        }
    }

    @Test
    void testSimplificationWithCSE() {
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

    // ============================================================================
    // Range-Based Index Simplification Tests
    // ============================================================================

    @Test
    void testDivisionByLargerConstant() {
        // Test: for %i in [0, 3) { %0 = %i / 3 } should simplify to %0 = 0
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        // %i / 3 when %i ∈ [0, 3) should be 0
        IndexExpr divExpr = IndexBinary.divide(i, new IndexConst(3));
        ScalarFromIndex fromIndex = new ScalarFromIndex(divExpr);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i", 3, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (i / 3) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should simplify to 0
        assertTrue(result.contains("from_index i64 0"), "i / 3 should simplify to 0");
        assertFalse(result.contains("/ 3"), "Division should be eliminated");
    }

    @Test
    void testModuloByLargerConstant() {
        // Test: for %i in [0, 3) { %0 = %i % 3 } should simplify to %0 = %i
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        // %i % 3 when %i ∈ [0, 3) should be %i
        IndexExpr modExpr = IndexBinary.modulo(i, new IndexConst(3));
        ScalarFromIndex fromIndex = new ScalarFromIndex(modExpr);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i", 3, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (i % 3) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should simplify to just %i
        assertTrue(result.contains("from_index i64 %i"), "i % 3 should simplify to i");
        assertFalse(result.contains("% 3"), "Modulo should be eliminated");
    }

    @Test
    void testBitwiseAndWithZero() {
        // Test: 0 & x → 0
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        // 0 & 1 should be 0
        IndexExpr andExpr =
                new IndexBinary(
                        IndexBinary.IndexBinaryOp.BITWISE_AND,
                        new IndexConst(0),
                        new IndexConst(1));
        ScalarFromIndex fromIndex = new ScalarFromIndex(andExpr);
        Store store = new Store(out, new IndexConst(0), fromIndex);

        LIRGraph graph = builder.build(store);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (0 & 1) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should simplify to 0
        assertTrue(result.contains("from_index i64 0"), "0 & 1 should simplify to 0");
        assertFalse(result.contains("&"), "Bitwise AND should be eliminated");
    }

    @Test
    void testChainedModuloSimplification() {
        // Test: for %i in [0, 3) { (%i % 3) % 6 } should simplify to %i
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        // First % 3, then % 6 - both should be eliminated
        IndexExpr innerMod = IndexBinary.modulo(i, new IndexConst(3));
        IndexExpr outerMod = IndexBinary.modulo(innerMod, new IndexConst(6));
        ScalarFromIndex fromIndex = new ScalarFromIndex(outerMod);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i", 3, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original ((i % 3) % 6) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should simplify to just %i
        assertTrue(result.contains("from_index i64 %i"), "Chained modulo should simplify to i");
        assertFalse(result.contains("% 3"), "First modulo should be eliminated");
        assertFalse(result.contains("% 6"), "Second modulo should be eliminated");
    }

    @Test
    void testNestedLoopsDifferentRanges() {
        // Test: for %i in [0, 3) { for %j in [0, 5) { i / 3, j / 5 } }
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        IndexVar j = new IndexVar("j");

        // i / 3 when i ∈ [0, 3) should be 0
        IndexExpr iDiv = IndexBinary.divide(i, new IndexConst(3));
        // j / 5 when j ∈ [0, 5) should be 0
        IndexExpr jDiv = IndexBinary.divide(j, new IndexConst(5));
        // Sum: 0 + 0 = 0
        IndexExpr sum = IndexBinary.add(iDiv, jDiv);

        ScalarFromIndex fromIndex = new ScalarFromIndex(sum);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop innerLoop = Loop.sequential("j", 5, store);
        Loop outerLoop = Loop.sequential("i", 3, innerLoop);

        LIRGraph graph = builder.build(outerLoop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (nested loops) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should simplify both divisions to 0, then 0 + 0 = 0
        assertTrue(result.contains("from_index i64 0"), "Both divisions should simplify to 0");
        assertFalse(result.contains("/ 3"), "i / 3 should be eliminated");
        assertFalse(result.contains("/ 5"), "j / 5 should be eliminated");
    }

    @Test
    void testDivisionNotSimplifiedWhenRangeTooLarge() {
        // Test: for %i in [0, 10) { %i / 3 } should NOT simplify (10 > 3)
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        // %i / 3 when %i ∈ [0, 10) should NOT be simplified
        IndexExpr divExpr = IndexBinary.divide(i, new IndexConst(3));
        ScalarFromIndex fromIndex = new ScalarFromIndex(divExpr);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i", 10, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (i / 3 with range 10) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should NOT simplify since range (10) > divisor (3)
        assertTrue(result.contains("/ 3"), "i / 3 should NOT be simplified when range > divisor");
    }

    @Test
    void testModuloNotSimplifiedWhenRangeTooLarge() {
        // Test: for %i in [0, 10) { %i % 3 } should NOT simplify to %i
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i = new IndexVar("i");
        // %i % 3 when %i ∈ [0, 10) should NOT be simplified to %i
        IndexExpr modExpr = IndexBinary.modulo(i, new IndexConst(3));
        ScalarFromIndex fromIndex = new ScalarFromIndex(modExpr);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i", 10, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (i % 3 with range 10) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Should NOT simplify since range (10) > divisor (3)
        assertTrue(result.contains("% 3"), "i % 3 should NOT be simplified when range > divisor");
    }

    @Test
    void testComplexExpressionWithUserExample() {
        // Test a complex expression similar to the user's example
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef out = builder.addContiguousOutput(DataType.I64, 1);

        IndexVar i2 = new IndexVar("i2");

        // Build: (%i2 / 3) & 1 should become 0 & 1 = 0
        IndexExpr i2Div3 = IndexBinary.divide(i2, new IndexConst(3));
        IndexExpr andExpr =
                new IndexBinary(IndexBinary.IndexBinaryOp.BITWISE_AND, i2Div3, new IndexConst(1));

        // And: %i2 % 3 % 6 should become %i2
        IndexExpr i2Mod3 = IndexBinary.modulo(i2, new IndexConst(3));
        IndexExpr i2Mod6 = IndexBinary.modulo(i2Mod3, new IndexConst(6));

        // Combine: andExpr + i2Mod6
        IndexExpr combined = IndexBinary.add(andExpr, i2Mod6);
        ScalarFromIndex fromIndex = new ScalarFromIndex(combined);
        Store store = new Store(out, new IndexConst(0), fromIndex);
        Loop loop = Loop.sequential("i2", 3, store);

        LIRGraph graph = builder.build(loop);

        // Print original
        LIRTextRenderer renderer = new LIRTextRenderer();
        String original = renderer.render(graph);
        System.out.println("=== Original (complex expression) ===");
        System.out.println(original);

        // Apply simplification
        IndexSimplificationPass pass = new IndexSimplificationPass();
        LIRGraph simplified = pass.run(graph);

        String result = renderer.render(simplified);
        System.out.println("=== Simplified ===");
        System.out.println(result);

        // Key assertions:
        // 1. i2 / 3 & 1 should become 0 (since i2/3=0, then 0&1=0)
        // 2. i2 % 3 % 6 should become i2
        assertTrue(result.contains("%i2"), "Should still reference i2");
        assertFalse(result.contains("/ 3"), "i2 / 3 should be eliminated");
    }
}

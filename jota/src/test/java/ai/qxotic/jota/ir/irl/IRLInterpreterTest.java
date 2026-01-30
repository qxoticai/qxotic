package ai.qxotic.jota.ir.irl;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.irt.BinaryOperator;
import ai.qxotic.jota.ir.irt.UnaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class IRLInterpreterTest {

    @Test
    void testSimpleElementwiseAdd() {
        // Test: output[i] = input0[i] + input1[i]
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            // Allocate buffers
            MemorySegment input0 = arena.allocate(size * Float.BYTES);
            MemorySegment input1 = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            // Initialize inputs
            for (int i = 0; i < size; i++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, i, i * 2.0f);
            }

            // Build IR-L graph
            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef in1 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            // Create loop body: output[i] = input0[i] + input1[i]
            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            ScalarExpr v1 = new ScalarLoad(in1, offset);
            ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, v0, v1);
            Store store = new Store(out, offset, sum);

            Loop loop = Loop.parallel("i", size, store);
            IRLGraph graph = builder.build(loop);

            // Execute
            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input0);
            interpreter.bindBuffer(in1.id(), input1);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            // Verify results
            for (int idx = 0; idx < size; idx++) {
                float expected = (idx + 1.0f) + (idx * 2.0f);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + idx);
            }
        }
    }

    @Test
    void testUnaryNegate() {
        // Test: output[i] = -input[i]
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            ScalarExpr neg = new ScalarUnary(UnaryOperator.NEGATE, v0);
            Store store = new Store(out, offset, neg);

            Loop loop = Loop.parallel("i", size, store);
            IRLGraph graph = builder.build(loop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = -(idx + 1.0f);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f);
            }
        }
    }

    @Test
    void testTernarySelect() {
        // Test: output[i] = cond[i] ? trueVal[i] : falseVal[i]
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment cond = arena.allocate(size);
            MemorySegment trueVal = arena.allocate(size * Float.BYTES);
            MemorySegment falseVal = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                cond.set(ValueLayout.JAVA_BYTE, i, (byte) (i % 2)); // alternating
                trueVal.setAtIndex(ValueLayout.JAVA_FLOAT, i, 100.0f + i);
                falseVal.setAtIndex(ValueLayout.JAVA_FLOAT, i, -100.0f - i);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef condBuf = builder.addContiguousInput(DataType.BOOL, size);
            BufferRef trueBuf = builder.addContiguousInput(DataType.FP32, size);
            BufferRef falseBuf = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr boolOffset = i;
            IndexExpr floatOffset = IndexBinary.mul(i, new IndexConst(Float.BYTES));

            ScalarExpr condVal = new ScalarLoad(condBuf, boolOffset);
            ScalarExpr trueExpr = new ScalarLoad(trueBuf, floatOffset);
            ScalarExpr falseExpr = new ScalarLoad(falseBuf, floatOffset);
            ScalarExpr select = new ScalarTernary(condVal, trueExpr, falseExpr);
            Store store = new Store(out, floatOffset, select);

            Loop loop = Loop.parallel("i", size, store);
            IRLGraph graph = builder.build(loop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(condBuf.id(), cond);
            interpreter.bindBuffer(trueBuf.id(), trueVal);
            interpreter.bindBuffer(falseBuf.id(), falseVal);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = (idx % 2 == 1) ? (100.0f + idx) : (-100.0f - idx);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + idx);
            }
        }
    }

    @Test
    void testBlockMultipleStatements() {
        // Test: v = input0[i] + input1[i]; output[i] = v * 2
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input0 = arena.allocate(size * Float.BYTES);
            MemorySegment input1 = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 2.0f);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef in1 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            ScalarExpr v1 = new ScalarLoad(in1, offset);
            ScalarExpr sum = new ScalarBinary(BinaryOperator.ADD, v0, v1);
            ScalarExpr two = ScalarConst.ofFloat(2.0f);
            ScalarExpr result = new ScalarBinary(BinaryOperator.MULTIPLY, sum, two);
            Store store = new Store(out, offset, result);

            Loop loop = Loop.parallel("i", size, store);
            IRLGraph graph = builder.build(loop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input0);
            interpreter.bindBuffer(in1.id(), input1);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = ((idx + 1.0f) + (idx + 2.0f)) * 2.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f);
            }
        }
    }

    @Test
    void testReductionSum() {
        // Test: acc = sum(input[i])
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float expectedSum = 0;
            for (int i = 0; i < size; i++) {
                float val = i + 1.0f;
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
                expectedSum += val;
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, 1);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);

            // Block: declare accumulator, update accumulator
            Accumulator acc = Accumulator.sum("acc", DataType.FP32);
            AccumulatorUpdate update = new AccumulatorUpdate("acc", v0);
            Block loopBody = Block.of(update);

            // After loop: read accumulator and store
            AccumulatorRead read = new AccumulatorRead("acc", DataType.FP32);

            // Create full program: declare acc, loop, store result
            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);

            // Execute manually since we need to handle accumulator declaration
            interpreter.executeNode(acc);
            for (int idx = 0; idx < size; idx++) {
                long offsetVal = (long) idx * Float.BYTES;
                MemorySegment buf = input;
                float val = buf.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offsetVal);
                ScalarConst constVal = ScalarConst.ofFloat(val);
                interpreter.executeNode(new AccumulatorUpdate("acc", constVal));
            }

            long accBits = interpreter.getAccumulatorValue("acc");
            float result = Float.intBitsToFloat((int) accBits);
            assertEquals(expectedSum, result, 1e-6f);
        }
    }

    @Test
    void testTiledLoop() {
        // Test: tiled copy with tile size 2
        try (Arena arena = Arena.ofConfined()) {
            int size = 8;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i * 10.0f);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            Store store = new Store(out, offset, v0);

            TiledLoop tiledLoop = TiledLoop.of("tile", "i", size, 2, store);
            IRLGraph graph = builder.build(tiledLoop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = idx * 10.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f);
            }
        }
    }

    @Test
    void testLoopNest() {
        // Test: 2D copy - output[i,j] = input[i,j]
        try (Arena arena = Arena.ofConfined()) {
            int rows = 2;
            int cols = 3;
            int size = rows * cols;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, rows, cols);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, rows, cols);

            IndexVar i = new IndexVar("i");
            IndexVar j = new IndexVar("j");
            // offset = (i * cols + j) * sizeof(float)
            IndexExpr linearIdx = IndexBinary.add(IndexBinary.mul(i, new IndexConst(cols)), j);
            IndexExpr offset = IndexBinary.mul(linearIdx, new IndexConst(Float.BYTES));

            ScalarExpr v0 = new ScalarLoad(in0, offset);
            Store store = new Store(out, offset, v0);

            Loop innerLoop = Loop.parallel("j", cols, store);
            Loop outerLoop = Loop.parallel("i", rows, innerLoop);

            IRLGraph graph = builder.build(outerLoop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = idx + 1.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f);
            }
        }
    }

    @Test
    void testScalarCast() {
        // Test: output[i] = (float) input[i] where input is int
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input = arena.allocate(size * Integer.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_INT, i, i + 10);
            }

            IRLGraph.Builder builder = IRLGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.I32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr inOffset = IndexBinary.mul(i, new IndexConst(Integer.BYTES));
            IndexExpr outOffset = IndexBinary.mul(i, new IndexConst(Float.BYTES));

            ScalarExpr v0 = new ScalarLoad(in0, inOffset);
            ScalarExpr cast = new ScalarCast(v0, DataType.FP32);
            Store store = new Store(out, outOffset, cast);

            Loop loop = Loop.parallel("i", size, store);
            IRLGraph graph = builder.build(loop);

            IRLInterpreter interpreter = new IRLInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            for (int idx = 0; idx < size; idx++) {
                float expected = (float) (idx + 10);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f);
            }
        }
    }

    @Test
    void testIndexExpressions() {
        IRLInterpreter interpreter = new IRLInterpreter();

        // Test constant
        assertEquals(42L, interpreter.evaluateIndex(new IndexConst(42)));

        // Test binary ops
        IndexConst a = new IndexConst(10);
        IndexConst b = new IndexConst(3);

        assertEquals(13L, interpreter.evaluateIndex(IndexBinary.add(a, b)));
        assertEquals(7L, interpreter.evaluateIndex(IndexBinary.sub(a, b)));
        assertEquals(30L, interpreter.evaluateIndex(IndexBinary.mul(a, b)));
        assertEquals(3L, interpreter.evaluateIndex(IndexBinary.div(a, b)));
        assertEquals(1L, interpreter.evaluateIndex(IndexBinary.mod(a, b)));
    }

    @Test
    void testScalarConstants() {
        IRLInterpreter interpreter = new IRLInterpreter();

        ScalarConst floatConst = ScalarConst.ofFloat(3.14f);
        assertEquals(
                3.14f, Float.intBitsToFloat((int) interpreter.evaluateScalar(floatConst)), 1e-6f);

        ScalarConst doubleConst = ScalarConst.ofDouble(2.718);
        assertEquals(
                2.718, Double.longBitsToDouble(interpreter.evaluateScalar(doubleConst)), 1e-10);

        ScalarConst intConst = ScalarConst.ofInt(42);
        assertEquals(42, (int) interpreter.evaluateScalar(intConst));

        ScalarConst longConst = ScalarConst.ofLong(123456789L);
        assertEquals(123456789L, interpreter.evaluateScalar(longConst));

        ScalarConst trueConst = ScalarConst.ofBool(true);
        assertEquals(1L, interpreter.evaluateScalar(trueConst));

        ScalarConst falseConst = ScalarConst.ofBool(false);
        assertEquals(0L, interpreter.evaluateScalar(falseConst));
    }
}

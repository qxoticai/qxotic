package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class LIRInterpreterTest {

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
            LIRGraph.Builder builder = LIRGraph.builder();
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
            LIRGraph graph = builder.build(loop);

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            ScalarExpr neg = new ScalarUnary(UnaryOperator.NEGATE, v0);
            Store store = new Store(out, offset, neg);

            Loop loop = Loop.parallel("i", size, store);
            LIRGraph graph = builder.build(loop);

            LIRInterpreter interpreter = new LIRInterpreter();
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
    void testGelu() {
        // Test GELU: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        try (Arena arena = Arena.ofConfined()) {
            float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            int size = inputData.length;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputData[i]);
            }

            LIRGraph.Builder builder = LIRGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr x = new ScalarLoad(in0, offset);

            // Constants
            ScalarExpr c_0_044715 = ScalarConst.ofFloat(0.044715f);
            ScalarExpr c_sqrt_2_pi = ScalarConst.ofFloat(0.79788456f);
            ScalarExpr c_1 = ScalarConst.ofFloat(1.0f);
            ScalarExpr c_0_5 = ScalarConst.ofFloat(0.5f);

            // x^2 = x * x
            ScalarExpr x_squared = new ScalarBinary(BinaryOperator.MULTIPLY, x, x);
            // x^3 = x^2 * x
            ScalarExpr x_cubed = new ScalarBinary(BinaryOperator.MULTIPLY, x_squared, x);
            // 0.044715 * x^3
            ScalarExpr scaled_cubic =
                    new ScalarBinary(BinaryOperator.MULTIPLY, c_0_044715, x_cubed);
            // x + 0.044715 * x^3
            ScalarExpr inner_sum = new ScalarBinary(BinaryOperator.ADD, x, scaled_cubic);
            // sqrt(2/pi) * (x + 0.044715 * x^3)
            ScalarExpr scaled_inner =
                    new ScalarBinary(BinaryOperator.MULTIPLY, c_sqrt_2_pi, inner_sum);
            // tanh(...)
            ScalarExpr tanh_result = new ScalarUnary(UnaryOperator.TANH, scaled_inner);
            // 1 + tanh(...)
            ScalarExpr one_plus_tanh = new ScalarBinary(BinaryOperator.ADD, c_1, tanh_result);
            // x * (1 + tanh(...))
            ScalarExpr x_times_bracket =
                    new ScalarBinary(BinaryOperator.MULTIPLY, x, one_plus_tanh);
            // 0.5 * x * (1 + tanh(...))
            ScalarExpr gelu = new ScalarBinary(BinaryOperator.MULTIPLY, c_0_5, x_times_bracket);

            Store store = new Store(out, offset, gelu);

            Loop loop = Loop.parallel("i", size, store);
            LIRGraph graph = builder.build(loop);

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(in0.id(), input);
            interpreter.bindBuffer(out.id(), output);
            interpreter.execute(graph);

            // Verify results
            for (int idx = 0; idx < size; idx++) {
                float expected = geluReference(inputData[idx]);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + idx);
            }
        }
    }

    private static float geluReference(float x) {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return (float) (0.5 * x * (1 + Math.tanh(inner)));
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

            LIRGraph.Builder builder = LIRGraph.builder();
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
            LIRGraph graph = builder.build(loop);

            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
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
            LIRGraph graph = builder.build(loop);

            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
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
            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.FP32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr offset = IndexBinary.mul(i, new IndexConst(Float.BYTES));
            ScalarExpr v0 = new ScalarLoad(in0, offset);
            Store store = new Store(out, offset, v0);

            TiledLoop tiledLoop = TiledLoop.of("tile", "i", size, 2, store);
            LIRGraph graph = builder.build(tiledLoop);

            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
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

            LIRGraph graph = builder.build(outerLoop);

            LIRInterpreter interpreter = new LIRInterpreter();
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

            LIRGraph.Builder builder = LIRGraph.builder();
            BufferRef in0 = builder.addContiguousInput(DataType.I32, size);
            BufferRef out = builder.addContiguousOutput(DataType.FP32, size);

            IndexVar i = new IndexVar("i");
            IndexExpr inOffset = IndexBinary.mul(i, new IndexConst(Integer.BYTES));
            IndexExpr outOffset = IndexBinary.mul(i, new IndexConst(Float.BYTES));

            ScalarExpr v0 = new ScalarLoad(in0, inOffset);
            ScalarExpr cast = new ScalarCast(v0, DataType.FP32);
            Store store = new Store(out, outOffset, cast);

            Loop loop = Loop.parallel("i", size, store);
            LIRGraph graph = builder.build(loop);

            LIRInterpreter interpreter = new LIRInterpreter();
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
        LIRInterpreter interpreter = new LIRInterpreter();

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
        LIRInterpreter interpreter = new LIRInterpreter();

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

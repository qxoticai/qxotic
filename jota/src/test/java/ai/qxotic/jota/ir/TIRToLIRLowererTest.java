package ai.qxotic.jota.ir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRInterpreter;
import ai.qxotic.jota.ir.lir.LIRTextRenderer;
import ai.qxotic.jota.ir.lir.ScalarInput;
import ai.qxotic.jota.ir.tir.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.Test;

class TIRToLIRLowererTest {

    @Test
    void testUnaryNegate() {
        // TIR: output = -input
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            // Allocate buffers
            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            // Initialize input
            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            UnaryOp neg = new UnaryOp(UnaryOperator.NEGATE, in0);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), neg);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            // Print LIR for debugging
            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Execute LIR
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify results
            for (int i = 0; i < size; i++) {
                float expected = -(i + 1.0f);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testBinaryAdd() {
        // TIR: output = input0 + input1
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input0 = arena.allocate(size * Float.BYTES);
            MemorySegment input1 = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, i, i * 2.0f);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            TensorInput in1 = new TensorInput(1, DataType.FP32, layout);
            BinaryOp add = new BinaryOp(BinaryOperator.ADD, in0, in1);

            TIRGraph tirGraph = new TIRGraph(List.of(in0, in1), add);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input0);
            interpreter.bindBuffer(1, input1);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = (i + 1.0f) + (i * 2.0f);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testBinaryAddWithScalarBroadcast() {
        // TIR: output = input + 2.0 (scalar broadcast)
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);

            // Scalar constant broadcast to match input shape
            Shape shape = Shape.flat(size);
            ScalarConstant scalar =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(2.0f), DataType.FP32, shape);

            BinaryOp add = new BinaryOp(BinaryOperator.ADD, in0, scalar);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), add);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = (i + 1.0f) + 2.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testChainedOps() {
        // TIR: output = (input0 + input1) * 2.0
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input0 = arena.allocate(size * Float.BYTES);
            MemorySegment input1 = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input0.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
                input1.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 2.0f);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            TensorInput in1 = new TensorInput(1, DataType.FP32, layout);

            BinaryOp add = new BinaryOp(BinaryOperator.ADD, in0, in1);

            Shape shape = Shape.flat(size);
            ScalarConstant two =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(2.0f), DataType.FP32, shape);
            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, add, two);

            TIRGraph tirGraph = new TIRGraph(List.of(in0, in1), mul);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input0);
            interpreter.bindBuffer(1, input1);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = ((i + 1.0f) + (i + 2.0f)) * 2.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testTernaryWhere() {
        // TIR: output = cond ? trueVal : falseVal
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment cond = arena.allocate(size); // BOOL = 1 byte
            MemorySegment trueVal = arena.allocate(size * Float.BYTES);
            MemorySegment falseVal = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                cond.set(ValueLayout.JAVA_BYTE, i, (byte) (i % 2)); // 0, 1, 0, 1
                trueVal.setAtIndex(ValueLayout.JAVA_FLOAT, i, 100.0f + i);
                falseVal.setAtIndex(ValueLayout.JAVA_FLOAT, i, -100.0f - i);
            }

            // Build TIR graph
            Layout boolLayout = Layout.rowMajor(size);
            Layout floatLayout = Layout.rowMajor(size);

            TensorInput condIn = new TensorInput(0, DataType.BOOL, boolLayout);
            TensorInput trueIn = new TensorInput(1, DataType.FP32, floatLayout);
            TensorInput falseIn = new TensorInput(2, DataType.FP32, floatLayout);

            TernaryOp where = new TernaryOp(TernaryOperator.WHERE, condIn, trueIn, falseIn);

            TIRGraph tirGraph = new TIRGraph(List.of(condIn, trueIn, falseIn), where);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, cond);
            interpreter.bindBuffer(1, trueVal);
            interpreter.bindBuffer(2, falseVal);
            interpreter.bindBuffer(3, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = (i % 2 == 1) ? (100.0f + i) : (-100.0f - i);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testCast() {
        // TIR: output = (float) input (int to float)
        try (Arena arena = Arena.ofConfined()) {
            int size = 4;

            MemorySegment input = arena.allocate(size * Integer.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_INT, i, i + 10);
            }

            // Build TIR graph
            Layout intLayout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.I32, intLayout);
            CastOp cast = new CastOp(in0, DataType.FP32);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), cast);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = (float) (i + 10);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void test2DArray() {
        // TIR: output[i,j] = input[i,j] * 2.0
        try (Arena arena = Arena.ofConfined()) {
            int rows = 2;
            int cols = 3;
            int size = rows * cols;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);

            Shape shape = Shape.flat(rows, cols);
            ScalarConstant two =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(2.0f), DataType.FP32, shape);
            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, in0, two);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), mul);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = (i + 1.0f) * 2.0f;
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    void testGelu() {
        // TIR: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        try (Arena arena = Arena.ofConfined()) {
            float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            int size = inputData.length;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(size * Float.BYTES);

            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputData[i]);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            Shape shape = Shape.flat(size);
            TensorInput x = new TensorInput(0, DataType.FP32, layout);

            // Constants
            ScalarConstant c0_5 =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(0.5f), DataType.FP32, shape);
            ScalarConstant c1 =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(1.0f), DataType.FP32, shape);
            ScalarConstant cSqrt2Pi =
                    ScalarConstant.broadcast(
                            Float.floatToRawIntBits(0.79788456f), DataType.FP32, shape);
            ScalarConstant c0_044715 =
                    ScalarConstant.broadcast(
                            Float.floatToRawIntBits(0.044715f), DataType.FP32, shape);

            // x^2 = x * x
            BinaryOp xSquared = new BinaryOp(BinaryOperator.MULTIPLY, x, x);
            // x^3 = x^2 * x
            BinaryOp xCubed = new BinaryOp(BinaryOperator.MULTIPLY, xSquared, x);
            // 0.044715 * x^3
            BinaryOp scaledCubic = new BinaryOp(BinaryOperator.MULTIPLY, c0_044715, xCubed);
            // x + 0.044715 * x^3
            BinaryOp innerSum = new BinaryOp(BinaryOperator.ADD, x, scaledCubic);
            // sqrt(2/pi) * (x + 0.044715 * x^3)
            BinaryOp scaledInner = new BinaryOp(BinaryOperator.MULTIPLY, cSqrt2Pi, innerSum);
            // tanh(...)
            UnaryOp tanhResult = new UnaryOp(UnaryOperator.TANH, scaledInner);
            // 1 + tanh(...)
            BinaryOp onePlusTanh = new BinaryOp(BinaryOperator.ADD, c1, tanhResult);
            // x * (1 + tanh(...))
            BinaryOp xTimesBracket = new BinaryOp(BinaryOperator.MULTIPLY, x, onePlusTanh);
            // 0.5 * x * (1 + tanh(...))
            BinaryOp gelu = new BinaryOp(BinaryOperator.MULTIPLY, c0_5, xTimesBracket);

            TIRGraph tirGraph = new TIRGraph(List.of(x), gelu);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < size; i++) {
                float expected = geluReference(inputData[i]);
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-5f, "Mismatch at index " + i);
            }
        }
    }

    private static float geluReference(float x) {
        double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return (float) (0.5 * x * (1 + Math.tanh(inner)));
    }

    // ==================== Reduction Tests ====================

    @Test
    void testReductionSum1D() {
        // TIR: output = sum(input)
        try (Arena arena = Arena.ofConfined()) {
            int size = 5;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float expectedSum = 0;
            for (int i = 0; i < size; i++) {
                float val = i + 1.0f;
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
                expectedSum += val;
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), sum);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedSum, actual, 1e-5f);
        }
    }

    @Test
    void testReductionMax1D() {
        // TIR: output = max(input)
        try (Arena arena = Arena.ofConfined()) {
            float[] inputData = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f};
            int size = inputData.length;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float expectedMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputData[i]);
                expectedMax = Math.max(expectedMax, inputData[i]);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp max = new ReductionOp(ReductionOperator.MAX, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), max);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedMax, actual, 1e-5f);
        }
    }

    @Test
    void testReductionMin1D() {
        // TIR: output = min(input)
        try (Arena arena = Arena.ofConfined()) {
            float[] inputData = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f};
            int size = inputData.length;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float expectedMin = Float.POSITIVE_INFINITY;
            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputData[i]);
                expectedMin = Math.min(expectedMin, inputData[i]);
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp min = new ReductionOp(ReductionOperator.MIN, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), min);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedMin, actual, 1e-5f);
        }
    }

    @Test
    void testReductionProd1D() {
        // TIR: output = prod(input)
        try (Arena arena = Arena.ofConfined()) {
            float[] inputData = {1.0f, 2.0f, 3.0f, 4.0f};
            int size = inputData.length;

            MemorySegment input = arena.allocate(size * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float expectedProd = 1.0f;
            for (int i = 0; i < size; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputData[i]);
                expectedProd *= inputData[i];
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(size);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp prod = new ReductionOp(ReductionOperator.PROD, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), prod);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedProd, actual, 1e-5f);
        }
    }

    @Test
    void testReductionSum2DAxis0() {
        // TIR: output[j] = sum(input[:, j], axis=0) - sum over rows
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(cols * Float.BYTES);

            // Initialize input and compute expected
            float[] expectedSums = new float[cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float val = i * cols + j + 1.0f;
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, val);
                    expectedSums[j] += val;
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int j = 0; j < cols; j++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                assertEquals(expectedSums[j], actual, 1e-5f, "Mismatch at column " + j);
            }
        }
    }

    @Test
    void testReductionSum2DAxis1() {
        // TIR: output[i] = sum(input[i, :], axis=1) - sum over columns
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(rows * Float.BYTES);

            // Initialize input and compute expected
            float[] expectedSums = new float[rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float val = i * cols + j + 1.0f;
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, val);
                    expectedSums[i] += val;
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, in0, new int[] {1}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < rows; i++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expectedSums[i], actual, 1e-5f, "Mismatch at row " + i);
            }
        }
    }

    @Test
    void testReductionSum2DAllAxes() {
        // TIR: output = sum(input) - sum over all axes
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            // Initialize input and compute expected
            float expectedSum = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float val = i * cols + j + 1.0f;
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, val);
                    expectedSum += val;
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, in0, new int[] {0, 1}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedSum, actual, 1e-4f);
        }
    }

    @Test
    void testReductionMax2DAxis0() {
        // TIR: output[j] = max(input[:, j], axis=0) - max over rows
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(cols * Float.BYTES);

            // Input data with clear max per column
            float[][] data = {
                {1.0f, 5.0f, 2.0f, 8.0f},
                {4.0f, 2.0f, 7.0f, 3.0f},
                {3.0f, 9.0f, 1.0f, 6.0f}
            };

            float[] expectedMax = new float[cols];
            for (int j = 0; j < cols; j++) {
                expectedMax[j] = Float.NEGATIVE_INFINITY;
            }

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, data[i][j]);
                    expectedMax[j] = Math.max(expectedMax[j], data[i][j]);
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp max = new ReductionOp(ReductionOperator.MAX, in0, new int[] {0}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), max);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify: expected max per column is [4, 9, 7, 8]
            for (int j = 0; j < cols; j++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                assertEquals(expectedMax[j], actual, 1e-5f, "Mismatch at column " + j);
            }
        }
    }

    @Test
    void testReductionMin2DAxis1() {
        // TIR: output[i] = min(input[i, :], axis=1) - min over columns
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(rows * Float.BYTES);

            // Input data with clear min per row
            float[][] data = {
                {5.0f, 2.0f, 8.0f, 3.0f}, // min = 2
                {7.0f, 4.0f, 1.0f, 9.0f}, // min = 1
                {6.0f, 3.0f, 5.0f, 2.0f} // min = 2
            };

            float[] expectedMin = new float[rows];
            for (int i = 0; i < rows; i++) {
                expectedMin[i] = Float.POSITIVE_INFINITY;
            }

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, data[i][j]);
                    expectedMin[i] = Math.min(expectedMin[i], data[i][j]);
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp min = new ReductionOp(ReductionOperator.MIN, in0, new int[] {1}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), min);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            // Verify: expected min per row is [2, 1, 2]
            for (int i = 0; i < rows; i++) {
                float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expectedMin[i], actual, 1e-5f, "Mismatch at row " + i);
            }
        }
    }

    @Test
    void testReductionMax2DAllAxes() {
        // TIR: output = max(input) - max over all axes
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float[][] data = {
                {1.0f, 5.0f, 2.0f, 8.0f},
                {4.0f, 2.0f, 7.0f, 3.0f},
                {3.0f, 9.0f, 1.0f, 6.0f} // 9 is the global max
            };

            float expectedMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, data[i][j]);
                    expectedMax = Math.max(expectedMax, data[i][j]);
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp max = new ReductionOp(ReductionOperator.MAX, in0, new int[] {0, 1}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), max);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedMax, actual, 1e-5f);
        }
    }

    @Test
    void testReductionMin2DAllAxes() {
        // TIR: output = min(input) - min over all axes
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            int cols = 4;

            MemorySegment input = arena.allocate(rows * cols * Float.BYTES);
            MemorySegment output = arena.allocate(Float.BYTES);

            float[][] data = {
                {5.0f, 2.0f, 8.0f, 3.0f},
                {7.0f, 4.0f, 1.0f, 9.0f}, // 1 is the global min
                {6.0f, 3.0f, 5.0f, 2.0f}
            };

            float expectedMin = Float.POSITIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    input.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, data[i][j]);
                    expectedMin = Math.min(expectedMin, data[i][j]);
                }
            }

            // Build TIR graph
            Layout layout = Layout.rowMajor(rows, cols);
            TensorInput in0 = new TensorInput(0, DataType.FP32, layout);
            ReductionOp min = new ReductionOp(ReductionOperator.MIN, in0, new int[] {0, 1}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(in0), min);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, input);
            interpreter.bindBuffer(1, output);
            interpreter.execute(lirGraph);

            float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            assertEquals(expectedMin, actual, 1e-5f);
        }
    }

    // ==================== Matmul Tests (tinygrad-style broadcast) ====================

    @Test
    void testMatmulBroadcast() {
        // Matmul via broadcast: C = A @ B
        // A (M, K), B (K, N) -> C (M, N)
        //
        // tinygrad approach:
        //   A.reshape(M, 1, K) * B.T.reshape(1, N, K) -> (M, N, K)
        //   .sum(axis=2) -> (M, N)
        //
        // C[i,j] = sum_k(A[i,k] * B[k,j])
        try (Arena arena = Arena.ofConfined()) {
            int M = 2;
            int K = 3;
            int N = 4;

            MemorySegment inputA = arena.allocate(M * K * Float.BYTES);
            MemorySegment inputB = arena.allocate(K * N * Float.BYTES);
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            // A = [[1, 2, 3],
            //      [4, 5, 6]]  (2x3)
            float[][] A = {{1, 2, 3}, {4, 5, 6}};
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i * K + k, A[i][k]);
                }
            }

            // B = [[1, 2, 3, 4],
            //      [5, 6, 7, 8],
            //      [9, 10, 11, 12]]  (3x4)
            float[][] B = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    inputB.setAtIndex(ValueLayout.JAVA_FLOAT, k * N + n, B[k][n]);
                }
            }

            // Compute expected C = A @ B
            float[][] expectedC = new float[M][N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        expectedC[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            // Build TIR graph
            // Input A: (M, K) row-major
            Layout layoutA = Layout.rowMajor(M, K);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            // Input B: (K, N) row-major
            Layout layoutB = Layout.rowMajor(K, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(M, 1, K): shape (M, 1, K), strides (K, 0, 1)
            // This broadcasts A over the N dimension
            Shape shapeA3D = Shape.flat(M, 1, K);
            Stride strideA3D = Stride.flat(K, 0, 1);
            Layout layoutA3D = Layout.of(shapeA3D, strideA3D);
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), shapeA3D);
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, K): first transpose B to (N, K), then broadcast to (1, N, K)
            // B[k,n] at offset k*N + n
            // B.T[n,k] = B[k,n] => strides (1, N) for shape (N, K)
            // Then broadcast to (1, N, K) with strides (0, 1, N)
            Shape shapeB3D = Shape.flat(1, N, K);
            Stride strideB3D = Stride.flat(0, 1, N);
            Layout layoutB3D = Layout.of(shapeB3D, strideB3D);
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), shapeB3D);
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            // Element-wise multiply: (M, 1, K) * (1, N, K) -> (M, N, K)
            // After broadcasting: (M, N, K) * (M, N, K)
            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);

            // Sum over axis 2 (K): (M, N, K) -> (M, N)
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-5f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulBroadcastLarger() {
        // Larger matmul: A (4, 5) @ B (5, 6) = C (4, 6)
        try (Arena arena = Arena.ofConfined()) {
            int M = 4;
            int K = 5;
            int N = 6;

            MemorySegment inputA = arena.allocate(M * K * Float.BYTES);
            MemorySegment inputB = arena.allocate(K * N * Float.BYTES);
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            // Initialize A with values 1..20
            float[][] A = new float[M][K];
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    A[i][k] = i * K + k + 1;
                    inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i * K + k, A[i][k]);
                }
            }

            // Initialize B with values 1..30
            float[][] B = new float[K][N];
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    B[k][n] = k * N + n + 1;
                    inputB.setAtIndex(ValueLayout.JAVA_FLOAT, k * N + n, B[k][n]);
                }
            }

            // Compute expected C = A @ B
            float[][] expectedC = new float[M][N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        expectedC[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            // Build TIR graph (same pattern as above)
            Layout layoutA = Layout.rowMajor(M, K);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            Layout layoutB = Layout.rowMajor(K, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(M, 1, K)
            Layout layoutA3D = Layout.of(Shape.flat(M, 1, K), Stride.flat(K, 0, 1));
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), layoutA3D.shape());
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, K)
            Layout layoutB3D = Layout.of(Shape.flat(1, N, K), Stride.flat(0, 1, N));
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), layoutB3D.shape());
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-4f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulSquare() {
        // Square matmul: A (3, 3) @ B (3, 3) = C (3, 3)
        try (Arena arena = Arena.ofConfined()) {
            int N = 3;

            MemorySegment inputA = arena.allocate(N * N * Float.BYTES);
            MemorySegment inputB = arena.allocate(N * N * Float.BYTES);
            MemorySegment output = arena.allocate(N * N * Float.BYTES);

            // A = identity matrix
            float[][] A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i * N + j, A[i][j]);
                }
            }

            // B = some matrix
            float[][] B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    inputB.setAtIndex(ValueLayout.JAVA_FLOAT, i * N + j, B[i][j]);
                }
            }

            // Expected: I @ B = B
            float[][] expectedC = B;

            // Build TIR graph
            Layout layoutA = Layout.rowMajor(N, N);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            Layout layoutB = Layout.rowMajor(N, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(N, 1, N)
            Layout layoutA3D = Layout.of(Shape.flat(N, 1, N), Stride.flat(N, 0, 1));
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), layoutA3D.shape());
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, N)
            Layout layoutB3D = Layout.of(Shape.flat(1, N, N), Stride.flat(0, 1, N));
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), layoutB3D.shape());
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-5f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulIota() {
        // Matmul with iota-initialized matrices:
        // A = iota(6).view(2, 3) = [[0, 1, 2], [3, 4, 5]]
        // B = iota(15).view(3, 5) = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
        // C = A @ B = (2, 5)
        try (Arena arena = Arena.ofConfined()) {
            int M = 2;
            int K = 3;
            int N = 5;

            MemorySegment inputA = arena.allocate(M * K * Float.BYTES);
            MemorySegment inputB = arena.allocate(K * N * Float.BYTES);
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            // A = iota(6).view(2, 3)
            // [[0, 1, 2],
            //  [3, 4, 5]]
            float[][] A = new float[M][K];
            for (int i = 0; i < M * K; i++) {
                A[i / K][i % K] = i;
                inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            // B = iota(15).view(3, 5)
            // [[0, 1, 2, 3, 4],
            //  [5, 6, 7, 8, 9],
            //  [10, 11, 12, 13, 14]]
            float[][] B = new float[K][N];
            for (int i = 0; i < K * N; i++) {
                B[i / N][i % N] = i;
                inputB.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            // Compute expected C = A @ B
            // C[0,0] = 0*0 + 1*5 + 2*10 = 25
            // C[0,1] = 0*1 + 1*6 + 2*11 = 28
            // ...
            float[][] expectedC = new float[M][N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        expectedC[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            // Build TIR graph
            Layout layoutA = Layout.rowMajor(M, K);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            Layout layoutB = Layout.rowMajor(K, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(M, 1, K)
            Layout layoutA3D = Layout.of(Shape.flat(M, 1, K), Stride.flat(K, 0, 1));
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), layoutA3D.shape());
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, K)
            Layout layoutB3D = Layout.of(Shape.flat(1, N, K), Stride.flat(0, 1, N));
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), layoutB3D.shape());
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            // Expected: [[25, 28, 31, 34, 37], [70, 82, 94, 106, 118]]
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-4f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulIotaLarger() {
        // Larger iota matmul:
        // A = iota(20).view(4, 5)
        // B = iota(30).view(5, 6)
        // C = A @ B = (4, 6)
        try (Arena arena = Arena.ofConfined()) {
            int M = 4;
            int K = 5;
            int N = 6;

            MemorySegment inputA = arena.allocate(M * K * Float.BYTES);
            MemorySegment inputB = arena.allocate(K * N * Float.BYTES);
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            // A = iota(20).view(4, 5)
            float[][] A = new float[M][K];
            for (int i = 0; i < M * K; i++) {
                A[i / K][i % K] = i;
                inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            // B = iota(30).view(5, 6)
            float[][] B = new float[K][N];
            for (int i = 0; i < K * N; i++) {
                B[i / N][i % N] = i;
                inputB.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            // Compute expected C = A @ B
            float[][] expectedC = new float[M][N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        expectedC[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            // Build TIR graph
            Layout layoutA = Layout.rowMajor(M, K);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            Layout layoutB = Layout.rowMajor(K, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(M, 1, K)
            Layout layoutA3D = Layout.of(Shape.flat(M, 1, K), Stride.flat(K, 0, 1));
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), layoutA3D.shape());
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, K)
            Layout layoutB3D = Layout.of(Shape.flat(1, N, K), Stride.flat(0, 1, N));
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), layoutB3D.shape());
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-3f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulIotaSquare() {
        // Square iota matmul:
        // A = iota(16).view(4, 4)
        // B = iota(16).view(4, 4)
        // C = A @ B = (4, 4)
        try (Arena arena = Arena.ofConfined()) {
            int N = 4;

            MemorySegment inputA = arena.allocate(N * N * Float.BYTES);
            MemorySegment inputB = arena.allocate(N * N * Float.BYTES);
            MemorySegment output = arena.allocate(N * N * Float.BYTES);

            // A = B = iota(16).view(4, 4)
            // [[0, 1, 2, 3],
            //  [4, 5, 6, 7],
            //  [8, 9, 10, 11],
            //  [12, 13, 14, 15]]
            float[][] A = new float[N][N];
            float[][] B = new float[N][N];
            for (int i = 0; i < N * N; i++) {
                A[i / N][i % N] = i;
                B[i / N][i % N] = i;
                inputA.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
                inputB.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            // Compute expected C = A @ B (A squared)
            float[][] expectedC = new float[N][N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        expectedC[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            // Build TIR graph
            Layout layoutA = Layout.rowMajor(N, N);
            TensorInput inA = new TensorInput(0, DataType.FP32, layoutA);

            Layout layoutB = Layout.rowMajor(N, N);
            TensorInput inB = new TensorInput(1, DataType.FP32, layoutB);

            // A.reshape(N, 1, N)
            Layout layoutA3D = Layout.of(Shape.flat(N, 1, N), Stride.flat(N, 0, 1));
            ViewKind kindA = new ViewKind.Broadcast(layoutA.shape(), layoutA3D.shape());
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, N)
            Layout layoutB3D = Layout.of(Shape.flat(1, N, N), Stride.flat(0, 1, N));
            ViewKind kindB = new ViewKind.Broadcast(layoutB.shape(), layoutB3D.shape());
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower and execute
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindBuffer(0, inputA);
            interpreter.bindBuffer(1, inputB);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify
            // Expected for iota(16)^2:
            // [[56, 62, 68, 74],
            //  [152, 174, 196, 218],
            //  [248, 286, 324, 362],
            //  [344, 398, 452, 506]]
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(
                            expectedC[i][j], actual, 1e-4f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulWithScalarInputs() {
        // Matmul with two scalar inputs broadcast to matrices:
        // A = Tensor.scalar(2f).view(2, 3) -> 2x3 matrix filled with 2.0
        // B = Tensor.scalar(3f).view(3, 5) -> 3x5 matrix filled with 3.0
        // C = A @ B -> 2x5 matrix
        // C[i,j] = sum_k(A[i,k] * B[k,j]) = sum_k(2.0 * 3.0) = 3 * 6.0 = 18.0
        try (Arena arena = Arena.ofConfined()) {
            int M = 2;
            int K = 3;
            int N = 5;

            // Scalar inputs - single float values
            MemorySegment scalarA = arena.allocate(Float.BYTES);
            MemorySegment scalarB = arena.allocate(Float.BYTES);
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            scalarA.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 2.0f);
            scalarB.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 3.0f);

            // Build TIR graph
            // ScalarConstant broadcast to (M, K) with stride (0, 0)
            Shape shapeA = Shape.flat(M, K);
            ScalarConstant inA =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(2.0f), DataType.FP32, shapeA);

            // ScalarConstant broadcast to (K, N) with stride (0, 0)
            Shape shapeB = Shape.flat(K, N);
            ScalarConstant inB =
                    ScalarConstant.broadcast(Float.floatToRawIntBits(3.0f), DataType.FP32, shapeB);

            // A.reshape(M, 1, K) for broadcast matmul
            Shape shapeA3D = Shape.flat(M, 1, K);
            Stride strideA3D = Stride.flat(0, 0, 0); // All zeros - broadcast scalar
            Layout layoutA3D = Layout.of(shapeA3D, strideA3D);
            ViewKind kindA = new ViewKind.Broadcast(shapeA, shapeA3D);
            ViewTransform viewA = new ViewTransform(inA, kindA, layoutA3D, false);

            // B.T.reshape(1, N, K) for broadcast matmul
            Shape shapeB3D = Shape.flat(1, N, K);
            Stride strideB3D = Stride.flat(0, 0, 0); // All zeros - broadcast scalar
            Layout layoutB3D = Layout.of(shapeB3D, strideB3D);
            ViewKind kindB = new ViewKind.Broadcast(shapeB, shapeB3D);
            ViewTransform viewB = new ViewTransform(inB, kindB, layoutB3D, false);

            // Element-wise multiply and reduce
            BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);
            ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

            // Both scalars are inputs (dynamic parameters)
            TIRGraph tirGraph = new TIRGraph(List.of(inA, inB), sum);

            // Lower to LIR
            TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
            LIRGraph lirGraph = lowerer.lower(tirGraph);

            System.out.println(new LIRTextRenderer().render(lirGraph));

            // Verify we have scalar inputs
            assertEquals(2, lirGraph.inputs().size());
            assertInstanceOf(ScalarInput.class, lirGraph.inputs().get(0), "First input should be scalar");
            assertInstanceOf(ScalarInput.class, lirGraph.inputs().get(1), "Second input should be scalar");

            // Execute
            LIRInterpreter interpreter = new LIRInterpreter();
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(2.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32);
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify: C[i,j] = K * (2.0 * 3.0) = 3 * 6.0 = 18.0
            float expected = K * 2.0f * 3.0f;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(expected, actual, 1e-5f, "Mismatch at C[" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulWithScalarConstantFolding() {
        // Test folding of matmul with broadcasted scalar constants
        // A: 2x3 tensor filled with 2.0, B: 3x5 tensor filled with 3.0
        // Matmul result: each element = sum_k(2.0 * 3.0) = 3 * 6.0 = 18.0
        // Expected: 2x5 tensor filled with 18.0

        // Scalar constant A = 2.0
        ScalarConstant scalarA = ScalarConstant.of(Float.floatToRawIntBits(2.0f), DataType.FP32);

        // Broadcast A to (2, 3)
        Shape shapeA2D = Shape.flat(2, 3);
        Layout layoutA2D = Layout.rowMajor(shapeA2D);
        ViewKind kindA = new ViewKind.Broadcast(Shape.scalar(), shapeA2D);
        ViewTransform broadcastA = new ViewTransform(scalarA, kindA, layoutA2D, false);

        // Scalar constant B = 3.0
        ScalarConstant scalarB = ScalarConstant.of(Float.floatToRawIntBits(3.0f), DataType.FP32);

        // Broadcast B to (3, 5)
        Shape shapeB2D = Shape.flat(3, 5);
        Layout layoutB2D = Layout.rowMajor(shapeB2D);
        ViewKind kindB = new ViewKind.Broadcast(Shape.scalar(), shapeB2D);
        ViewTransform broadcastB = new ViewTransform(scalarB, kindB, layoutB2D, false);

        // Reshape A to (2, 1, 3) for matmul broadcasting
        Shape shapeA3D = Shape.flat(2, 1, 3);
        Layout layoutA3D = Layout.of(shapeA3D, Stride.flat(3, 0, 1));
        ViewKind reshapeA = new ViewKind.Broadcast(shapeA2D, shapeA3D);
        ViewTransform viewA = new ViewTransform(broadcastA, reshapeA, layoutA3D, false);

        // Reshape B.T to (1, 5, 3) for matmul broadcasting
        Shape shapeB3D = Shape.flat(1, 5, 3);
        Layout layoutB3D = Layout.of(shapeB3D, Stride.flat(0, 1, 5));
        ViewKind reshapeB = new ViewKind.Broadcast(shapeB2D, shapeB3D);
        ViewTransform viewB = new ViewTransform(broadcastB, reshapeB, layoutB3D, false);

        // Element-wise multiply: (2, 1, 3) * (1, 5, 3) -> (2, 5, 3)
        BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, viewA, viewB);

        // Sum over axis 2 (the contraction dimension K=3): (2, 5, 3) -> (2, 5)
        ReductionOp sum = new ReductionOp(ReductionOperator.SUM, mul, new int[] {2}, false);

        // Build TIR graph with no inputs (all constants)
        TIRGraph tirGraph = new TIRGraph(List.of(), sum);

        // Run constant folding pass
        TIRConstantFoldingPass foldingPass = new TIRConstantFoldingPass();
        TIRGraph foldedGraph = foldingPass.run(tirGraph);

        // Verify the result
        assertEquals(1, foldedGraph.outputs().size());
        TIRNode result = foldedGraph.outputs().get(0);

        // Result should be a ScalarConstant
        assertTrue(
                result instanceof ScalarConstant,
                "Expected ScalarConstant but got " + result.getClass().getSimpleName());
        ScalarConstant resultSc = (ScalarConstant) result;

        // Check shape: should be (2, 5)
        assertEquals(Shape.flat(2, 5), resultSc.layout().shape());

        // Check value: 2.0 * 3.0 * 3 = 18.0
        float expectedValue = 2.0f * 3.0f * 3; // a * b * K
        float actualValue = Float.intBitsToFloat((int) resultSc.rawBits());
        assertEquals(expectedValue, actualValue, 1e-5f);
    }
}

package ai.qxotic.jota.ir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRInterpreter;
import ai.qxotic.jota.ir.lir.LIRTextRenderer;
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
}

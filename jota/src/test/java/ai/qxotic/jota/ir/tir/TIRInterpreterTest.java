package ai.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.memory.*;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TIRInterpreterTest {

    private static MemoryContext<MemorySegment> context;
    private static MemoryAccess<MemorySegment> memAccess;

    @BeforeAll
    static void setUp() {
        context = ContextFactory.ofMemorySegment();
        memAccess = context.memoryAccess();
    }

    // ==================== Unary Operations ====================

    @Test
    void testUnaryNegate() {
        MemoryView<?> input = createFloatTensor(new float[] {-1.0f, 2.0f, -3.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode negateOp = new UnaryOp(UnaryOperator.NEGATE, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(negateOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(1, outputs.size());
        assertFloatEquals(new float[] {1.0f, -2.0f, 3.0f}, outputs.get(0));
    }

    @Test
    void testUnaryAbs() {
        MemoryView<?> input = createFloatTensor(new float[] {-1.0f, 2.0f, -3.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode absOp = new UnaryOp(UnaryOperator.ABS, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(absOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {1.0f, 2.0f, 3.0f}, outputs.get(0));
    }

    @Test
    void testUnaryExp() {
        MemoryView<?> input = createFloatTensor(new float[] {0.0f, 1.0f, 2.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode expOp = new UnaryOp(UnaryOperator.EXP, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(expOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(
                new float[] {1.0f, (float) Math.E, (float) Math.exp(2.0)}, outputs.get(0));
    }

    @Test
    void testUnarySqrt() {
        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 4.0f, 9.0f, 16.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode sqrtOp = new UnaryOp(UnaryOperator.SQRT, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(sqrtOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, outputs.get(0));
    }

    @Test
    void testUnaryReciprocal() {
        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 2.0f, 4.0f, 5.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode recipOp = new UnaryOp(UnaryOperator.RECIPROCAL, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(recipOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {1.0f, 0.5f, 0.25f, 0.2f}, outputs.get(0));
    }

    // ==================== Binary Operations ====================

    @Test
    void testBinaryAdd() {
        MemoryView<?> input1 = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {4.0f, 5.0f, 6.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {5.0f, 7.0f, 9.0f}, outputs.get(0));
    }

    @Test
    void testBinarySubtract() {
        MemoryView<?> input1 = createFloatTensor(new float[] {10.0f, 20.0f, 30.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {1.0f, 5.0f, 10.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode subOp = new BinaryOp(BinaryOperator.SUBTRACT, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(subOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {9.0f, 15.0f, 20.0f}, outputs.get(0));
    }

    @Test
    void testBinaryMultiply() {
        MemoryView<?> input1 = createFloatTensor(new float[] {2.0f, 3.0f, 4.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {5.0f, 6.0f, 7.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode mulOp = new BinaryOp(BinaryOperator.MULTIPLY, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(mulOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {10.0f, 18.0f, 28.0f}, outputs.get(0));
    }

    @Test
    void testBinaryDivide() {
        MemoryView<?> input1 = createFloatTensor(new float[] {10.0f, 20.0f, 30.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {2.0f, 4.0f, 5.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode divOp = new BinaryOp(BinaryOperator.DIVIDE, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(divOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {5.0f, 5.0f, 6.0f}, outputs.get(0));
    }

    @Test
    void testBinaryMin() {
        MemoryView<?> input1 = createFloatTensor(new float[] {1.0f, 5.0f, 3.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {2.0f, 4.0f, 6.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode minOp = new BinaryOp(BinaryOperator.MIN, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(minOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {1.0f, 4.0f, 3.0f}, outputs.get(0));
    }

    @Test
    void testBinaryMax() {
        MemoryView<?> input1 = createFloatTensor(new float[] {1.0f, 5.0f, 3.0f});
        MemoryView<?> input2 = createFloatTensor(new float[] {2.0f, 4.0f, 6.0f});

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode maxOp = new BinaryOp(BinaryOperator.MAX, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(maxOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertFloatEquals(new float[] {2.0f, 5.0f, 6.0f}, outputs.get(0));
    }

    // ==================== Scalar Constant ====================

    @Test
    void testScalarConstantAdd() {
        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode scalar = ScalarConstant.broadcast(floatBits(10.0f), DataType.FP32, Shape.of(3));
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput, scalar);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {11.0f, 12.0f, 13.0f}, outputs.get(0));
    }

    @Test
    void testScalarConstantMultiply() {
        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f, 4.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode scalar = ScalarConstant.broadcast(floatBits(2.0f), DataType.FP32, Shape.of(4));
        TIRNode mulOp = new BinaryOp(BinaryOperator.MULTIPLY, tensorInput, scalar);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(mulOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {2.0f, 4.0f, 6.0f, 8.0f}, outputs.get(0));
    }

    // ==================== Cast Operation ====================

    @Test
    void testCastFP32toFP64() {
        MemoryView<?> input = createFloatTensor(new float[] {1.5f, 2.5f, 3.5f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode castOp = new CastOp(tensorInput, DataType.FP64);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(castOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(DataType.FP64, outputs.get(0).dataType());
        assertDoubleEquals(new double[] {1.5, 2.5, 3.5}, outputs.get(0));
    }

    @Test
    void testCastFP32toI32() {
        MemoryView<?> input = createFloatTensor(new float[] {1.9f, 2.1f, -3.7f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode castOp = new CastOp(tensorInput, DataType.I32);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(castOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(DataType.I32, outputs.get(0).dataType());
        assertIntEquals(new int[] {1, 2, -3}, outputs.get(0));
    }

    @Test
    void testCastI32toFP32() {
        MemoryView<?> input = createIntTensor(new int[] {1, 2, 3, 4});

        TIRNode tensorInput = new TensorInput(0, DataType.I32, input.layout());
        TIRNode castOp = new CastOp(tensorInput, DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(castOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(DataType.FP32, outputs.get(0).dataType());
        assertFloatEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, outputs.get(0));
    }

    // ==================== Ternary Operation (WHERE) ====================

    @Test
    void testTernaryWhere() {
        MemoryView<?> cond = createBoolTensor(new boolean[] {true, false, true, false});
        MemoryView<?> trueVal = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
        MemoryView<?> falseVal = createFloatTensor(new float[] {10.0f, 20.0f, 30.0f, 40.0f});

        TIRNode condInput = new TensorInput(0, DataType.BOOL, cond.layout());
        TIRNode trueInput = new TensorInput(1, DataType.FP32, trueVal.layout());
        TIRNode falseInput = new TensorInput(2, DataType.FP32, falseVal.layout());
        TIRNode whereOp = new TernaryOp(TernaryOperator.WHERE, condInput, trueInput, falseInput);
        TIRGraph graph = new TIRGraph(List.of(condInput, trueInput, falseInput), List.of(whereOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(cond, trueVal, falseVal), context);

        assertFloatEquals(new float[] {1.0f, 20.0f, 3.0f, 40.0f}, outputs.get(0));
    }

    // ==================== Reduction Operations ====================

    @Test
    void testReductionSum() {
        MemoryView<?> input =
                createFloatTensor(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape.of(2, 3));

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        // Reduce along axis 1 (columns), keepDims=false -> output shape [2]
        TIRNode sumOp =
                new ReductionOp(
                        ReductionOperator.SUM, tensorInput, new int[] {1}, false, DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(sumOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(Shape.of(2), outputs.get(0).shape());
        assertFloatEquals(new float[] {6.0f, 15.0f}, outputs.get(0)); // 1+2+3=6, 4+5+6=15
    }

    @Test
    void testReductionMax() {
        MemoryView<?> input =
                createFloatTensor(new float[] {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f}, Shape.of(2, 3));

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode maxOp =
                new ReductionOp(
                        ReductionOperator.MAX, tensorInput, new int[] {1}, false, DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(maxOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(Shape.of(2), outputs.get(0).shape());
        assertFloatEquals(new float[] {5.0f, 6.0f}, outputs.get(0));
    }

    @Test
    void testReductionMin() {
        MemoryView<?> input =
                createFloatTensor(new float[] {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f}, Shape.of(2, 3));

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode minOp =
                new ReductionOp(
                        ReductionOperator.MIN, tensorInput, new int[] {1}, false, DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(minOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(Shape.of(2), outputs.get(0).shape());
        assertFloatEquals(new float[] {1.0f, 2.0f}, outputs.get(0));
    }

    // ==================== Iota Constant ====================

    @Test
    void testIotaConstant() {
        TIRNode iota = IotaConstant.of(5, DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(), List.of(iota));

        List<MemoryView<MemorySegment>> outputs = TIRInterpreter.execute(graph, List.of(), context);

        assertEquals(Shape.of(5), outputs.get(0).shape());
        assertFloatEquals(new float[] {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, outputs.get(0));
    }

    @Test
    void testIotaConstantI32() {
        TIRNode iota = IotaConstant.of(4, DataType.I32);
        TIRGraph graph = new TIRGraph(List.of(), List.of(iota));

        List<MemoryView<MemorySegment>> outputs = TIRInterpreter.execute(graph, List.of(), context);

        assertEquals(DataType.I32, outputs.get(0).dataType());
        assertIntEquals(new int[] {0, 1, 2, 3}, outputs.get(0));
    }

    // ==================== View Transform ====================

    @Test
    void testViewTransformBroadcast() {
        // Create a scalar constant and broadcast it
        TIRNode scalar = ScalarConstant.of(floatBits(5.0f), DataType.FP32);
        Layout broadcastLayout = Layout.of(Shape.of(3), Stride.zeros(Shape.of(3)));
        ViewKind broadcastKind = new ViewKind.Broadcast(Shape.scalar(), Shape.of(3));
        TIRNode broadcast = new ViewTransform(scalar, broadcastKind, broadcastLayout, false);

        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f});
        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput, broadcast);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {6.0f, 7.0f, 8.0f}, outputs.get(0));
    }

    // ==================== 2D Tensor Operations ====================

    @Test
    void testBinaryAdd2D() {
        MemoryView<?> input1 =
                createFloatTensor(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape.of(2, 3));
        MemoryView<?> input2 =
                createFloatTensor(
                        new float[] {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, Shape.of(2, 3));

        TIRNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertEquals(Shape.of(2, 3), outputs.get(0).shape());
        assertFloatEquals(new float[] {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}, outputs.get(0));
    }

    // ==================== Integer Operations ====================

    @Test
    void testBinaryAddI32() {
        MemoryView<?> input1 = createIntTensor(new int[] {1, 2, 3, 4});
        MemoryView<?> input2 = createIntTensor(new int[] {10, 20, 30, 40});

        TIRNode tensorInput1 = new TensorInput(0, DataType.I32, input1.layout());
        TIRNode tensorInput2 = new TensorInput(1, DataType.I32, input2.layout());
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput1, tensorInput2);
        TIRGraph graph = new TIRGraph(List.of(tensorInput1, tensorInput2), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input1, input2), context);

        assertIntEquals(new int[] {11, 22, 33, 44}, outputs.get(0));
    }

    @Test
    void testUnaryNegateI32() {
        MemoryView<?> input = createIntTensor(new int[] {1, -2, 3, -4});

        TIRNode tensorInput = new TensorInput(0, DataType.I32, input.layout());
        TIRNode negateOp = new UnaryOp(UnaryOperator.NEGATE, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(negateOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertIntEquals(new int[] {-1, 2, -3, 4}, outputs.get(0));
    }

    // ==================== Composite Operations ====================

    @Test
    void testChainedOperations() {
        // (a + b) * c
        MemoryView<?> a = createFloatTensor(new float[] {1.0f, 2.0f, 3.0f});
        MemoryView<?> b = createFloatTensor(new float[] {4.0f, 5.0f, 6.0f});
        MemoryView<?> c = createFloatTensor(new float[] {2.0f, 2.0f, 2.0f});

        TIRNode inputA = new TensorInput(0, DataType.FP32, a.layout());
        TIRNode inputB = new TensorInput(1, DataType.FP32, b.layout());
        TIRNode inputC = new TensorInput(2, DataType.FP32, c.layout());
        TIRNode addOp = new BinaryOp(BinaryOperator.ADD, inputA, inputB);
        TIRNode mulOp = new BinaryOp(BinaryOperator.MULTIPLY, addOp, inputC);
        TIRGraph graph = new TIRGraph(List.of(inputA, inputB, inputC), List.of(mulOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(a, b, c), context);

        // (1+4)*2=10, (2+5)*2=14, (3+6)*2=18
        assertFloatEquals(new float[] {10.0f, 14.0f, 18.0f}, outputs.get(0));
    }

    @Test
    void testMultipleOutputs() {
        MemoryView<?> input = createFloatTensor(new float[] {1.0f, 4.0f, 9.0f});

        TIRNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode sqrtOp = new UnaryOp(UnaryOperator.SQRT, tensorInput);
        TIRNode negateOp = new UnaryOp(UnaryOperator.NEGATE, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(sqrtOp, negateOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertEquals(2, outputs.size());
        assertFloatEquals(new float[] {1.0f, 2.0f, 3.0f}, outputs.get(0));
        assertFloatEquals(new float[] {-1.0f, -4.0f, -9.0f}, outputs.get(1));
    }

    // ==================== Double Precision ====================

    @Test
    void testUnaryExpFP64() {
        MemoryView<?> input = createDoubleTensor(new double[] {0.0, 1.0, 2.0});

        TIRNode tensorInput = new TensorInput(0, DataType.FP64, input.layout());
        TIRNode expOp = new UnaryOp(UnaryOperator.EXP, tensorInput);
        TIRGraph graph = new TIRGraph(List.of(tensorInput), List.of(expOp));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertDoubleEquals(new double[] {1.0, Math.E, Math.exp(2.0)}, outputs.get(0));
    }

    // ==================== Non-trivial Composite Operations ====================

    /**
     * Tests GELU activation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) This
     * exercises a complex expression graph with multiple operations.
     */
    @Test
    void testGelu() {
        float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());

        // Constants
        TIRNode c_0_044715 = ScalarConstant.broadcast(floatBits(0.044715f), DataType.FP32, shape);
        TIRNode c_sqrt_2_pi =
                ScalarConstant.broadcast(floatBits(0.7978845608f), DataType.FP32, shape);
        TIRNode c_1 = ScalarConstant.broadcast(floatBits(1.0f), DataType.FP32, shape);
        TIRNode c_0_5 = ScalarConstant.broadcast(floatBits(0.5f), DataType.FP32, shape);

        // x^2
        TIRNode x_squared = new BinaryOp(BinaryOperator.MULTIPLY, x, x);
        // x^3 = x^2 * x
        TIRNode x_cubed = new BinaryOp(BinaryOperator.MULTIPLY, x_squared, x);
        // 0.044715 * x^3
        TIRNode scaled_cubic = new BinaryOp(BinaryOperator.MULTIPLY, c_0_044715, x_cubed);
        // x + 0.044715 * x^3
        TIRNode inner_sum = new BinaryOp(BinaryOperator.ADD, x, scaled_cubic);
        // sqrt(2/pi) * (x + 0.044715 * x^3)
        TIRNode scaled_inner = new BinaryOp(BinaryOperator.MULTIPLY, c_sqrt_2_pi, inner_sum);
        // tanh(...)
        TIRNode tanh_result = new UnaryOp(UnaryOperator.TANH, scaled_inner);
        // 1 + tanh(...)
        TIRNode one_plus_tanh = new BinaryOp(BinaryOperator.ADD, c_1, tanh_result);
        // x * (1 + tanh(...))
        TIRNode x_times_bracket = new BinaryOp(BinaryOperator.MULTIPLY, x, one_plus_tanh);
        // 0.5 * x * (1 + tanh(...))
        TIRNode gelu = new BinaryOp(BinaryOperator.MULTIPLY, c_0_5, x_times_bracket);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(gelu));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        // Expected GELU values (computed externally)
        float[] expected = new float[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            expected[i] = geluReference(inputData[i]);
        }
        assertFloatEquals(expected, outputs.get(0));
    }

    /** Tests Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x)) */
    @Test
    void testSigmoid() {
        float[] inputData = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());

        // Constants
        TIRNode c_1 = ScalarConstant.broadcast(floatBits(1.0f), DataType.FP32, shape);

        // -x
        TIRNode neg_x = new UnaryOp(UnaryOperator.NEGATE, x);
        // exp(-x)
        TIRNode exp_neg_x = new UnaryOp(UnaryOperator.EXP, neg_x);
        // 1 + exp(-x)
        TIRNode one_plus_exp = new BinaryOp(BinaryOperator.ADD, c_1, exp_neg_x);
        // 1 / (1 + exp(-x))
        TIRNode sigmoid = new UnaryOp(UnaryOperator.RECIPROCAL, one_plus_exp);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(sigmoid));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        float[] expected = new float[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            expected[i] = (float) (1.0 / (1.0 + Math.exp(-inputData[i])));
        }
        assertFloatEquals(expected, outputs.get(0));
    }

    /** Tests SiLU (Swish) activation: silu(x) = x * sigmoid(x) */
    @Test
    void testSilu() {
        float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode c_1 = ScalarConstant.broadcast(floatBits(1.0f), DataType.FP32, shape);

        // sigmoid(x) = 1 / (1 + exp(-x))
        TIRNode neg_x = new UnaryOp(UnaryOperator.NEGATE, x);
        TIRNode exp_neg_x = new UnaryOp(UnaryOperator.EXP, neg_x);
        TIRNode one_plus_exp = new BinaryOp(BinaryOperator.ADD, c_1, exp_neg_x);
        TIRNode sigmoid = new UnaryOp(UnaryOperator.RECIPROCAL, one_plus_exp);

        // x * sigmoid(x)
        TIRNode silu = new BinaryOp(BinaryOperator.MULTIPLY, x, sigmoid);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(silu));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        float[] expected = new float[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            float sig = (float) (1.0 / (1.0 + Math.exp(-inputData[i])));
            expected[i] = inputData[i] * sig;
        }
        assertFloatEquals(expected, outputs.get(0));
    }

    /** Tests ReLU using max(x, 0). */
    @Test
    void testRelu() {
        float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());
        TIRNode zero = ScalarConstant.broadcast(floatBits(0.0f), DataType.FP32, shape);

        TIRNode relu = new BinaryOp(BinaryOperator.MAX, x, zero);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(relu));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        assertFloatEquals(new float[] {0.0f, 0.0f, 0.0f, 1.0f, 2.0f}, outputs.get(0));
    }

    /** Tests softmax-like computation: exp(x) / sum(exp(x)) For a single row to keep it simple. */
    @Test
    void testSoftmaxRow() {
        float[] inputData = {1.0f, 2.0f, 3.0f};
        MemoryView<?> input = createFloatTensor(inputData);

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());

        // exp(x)
        TIRNode exp_x = new UnaryOp(UnaryOperator.EXP, x);

        // sum(exp(x)) via reduction
        TIRNode sum_exp =
                new ReductionOp(ReductionOperator.SUM, exp_x, new int[] {0}, false, DataType.FP32);

        // Broadcast sum back to original shape for division
        Layout broadcastLayout = Layout.of(Shape.of(3), Stride.zeros(Shape.of(3)));
        ViewKind broadcastKind = new ViewKind.Broadcast(Shape.scalar(), Shape.of(3));
        TIRNode sum_broadcast = new ViewTransform(sum_exp, broadcastKind, broadcastLayout, false);

        // exp(x) / sum(exp(x))
        TIRNode softmax = new BinaryOp(BinaryOperator.DIVIDE, exp_x, sum_broadcast);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(softmax));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        // Compute expected softmax
        double[] exp = new double[3];
        double sum = 0;
        for (int i = 0; i < 3; i++) {
            exp[i] = Math.exp(inputData[i]);
            sum += exp[i];
        }
        float[] expected = new float[3];
        for (int i = 0; i < 3; i++) {
            expected[i] = (float) (exp[i] / sum);
        }

        assertFloatEquals(expected, outputs.get(0));

        // Verify softmax sums to 1
        float actualSum = 0;
        for (int i = 0; i < 3; i++) {
            actualSum += readFloat(outputs.get(0), i);
        }
        assertEquals(1.0f, actualSum, 0.0001f);
    }

    /** Tests layer normalization-like computation: (x - mean(x)) / sqrt(var(x) + eps) */
    @Test
    void testLayerNormLike() {
        float[] inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();
        int n = inputData.length;

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());

        // mean = sum(x) / n
        TIRNode sum_x =
                new ReductionOp(ReductionOperator.SUM, x, new int[] {0}, false, DataType.FP32);
        Layout scalarBroadcast = Layout.of(shape, Stride.zeros(shape));
        ViewKind broadcastKind = new ViewKind.Broadcast(Shape.scalar(), shape);
        TIRNode sum_broadcast = new ViewTransform(sum_x, broadcastKind, scalarBroadcast, false);
        TIRNode n_const = ScalarConstant.broadcast(floatBits((float) n), DataType.FP32, shape);
        TIRNode mean = new BinaryOp(BinaryOperator.DIVIDE, sum_broadcast, n_const);

        // x - mean
        TIRNode x_centered = new BinaryOp(BinaryOperator.SUBTRACT, x, mean);

        // (x - mean)^2
        TIRNode x_centered_sq = new BinaryOp(BinaryOperator.MULTIPLY, x_centered, x_centered);

        // var = sum((x - mean)^2) / n
        TIRNode sum_sq =
                new ReductionOp(
                        ReductionOperator.SUM, x_centered_sq, new int[] {0}, false, DataType.FP32);
        TIRNode sum_sq_broadcast = new ViewTransform(sum_sq, broadcastKind, scalarBroadcast, false);
        TIRNode var = new BinaryOp(BinaryOperator.DIVIDE, sum_sq_broadcast, n_const);

        // sqrt(var + eps)
        TIRNode eps = ScalarConstant.broadcast(floatBits(1e-5f), DataType.FP32, shape);
        TIRNode var_plus_eps = new BinaryOp(BinaryOperator.ADD, var, eps);
        TIRNode std = new UnaryOp(UnaryOperator.SQRT, var_plus_eps);

        // (x - mean) / std
        TIRNode normalized = new BinaryOp(BinaryOperator.DIVIDE, x_centered, std);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(normalized));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        // Compute expected normalized values
        double meanVal = 0;
        for (float v : inputData) meanVal += v;
        meanVal /= n;

        double varVal = 0;
        for (float v : inputData) varVal += (v - meanVal) * (v - meanVal);
        varVal /= n;

        double stdVal = Math.sqrt(varVal + 1e-5);

        float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            expected[i] = (float) ((inputData[i] - meanVal) / stdVal);
        }

        assertFloatEquals(expected, outputs.get(0));

        // Verify normalized values have mean ≈ 0
        float actualMean = 0;
        for (int i = 0; i < n; i++) {
            actualMean += readFloat(outputs.get(0), i);
        }
        actualMean /= n;
        assertEquals(0.0f, actualMean, 0.0001f);
    }

    /** Tests a polynomial: y = 2x^3 - 3x^2 + x - 5 */
    @Test
    void testPolynomial() {
        float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        MemoryView<?> input = createFloatTensor(inputData);
        Shape shape = input.shape();

        TIRNode x = new TensorInput(0, DataType.FP32, input.layout());

        // Constants
        TIRNode c_2 = ScalarConstant.broadcast(floatBits(2.0f), DataType.FP32, shape);
        TIRNode c_3 = ScalarConstant.broadcast(floatBits(3.0f), DataType.FP32, shape);
        TIRNode c_5 = ScalarConstant.broadcast(floatBits(5.0f), DataType.FP32, shape);

        // x^2
        TIRNode x2 = new BinaryOp(BinaryOperator.MULTIPLY, x, x);
        // x^3
        TIRNode x3 = new BinaryOp(BinaryOperator.MULTIPLY, x2, x);
        // 2x^3
        TIRNode term1 = new BinaryOp(BinaryOperator.MULTIPLY, c_2, x3);
        // 3x^2
        TIRNode term2 = new BinaryOp(BinaryOperator.MULTIPLY, c_3, x2);
        // 2x^3 - 3x^2
        TIRNode sum1 = new BinaryOp(BinaryOperator.SUBTRACT, term1, term2);
        // 2x^3 - 3x^2 + x
        TIRNode sum2 = new BinaryOp(BinaryOperator.ADD, sum1, x);
        // 2x^3 - 3x^2 + x - 5
        TIRNode result = new BinaryOp(BinaryOperator.SUBTRACT, sum2, c_5);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(result));

        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, List.of(input), context);

        float[] expected = new float[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            float v = inputData[i];
            expected[i] = 2 * v * v * v - 3 * v * v + v - 5;
        }
        assertFloatEquals(expected, outputs.get(0));
    }

    private static float geluReference(float x) {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return (float) (0.5 * x * (1 + Math.tanh(inner)));
    }

    // ==================== Helper Methods ====================

    private MemoryView<?> createFloatTensor(float[] data) {
        return createFloatTensor(data, Shape.of(data.length));
    }

    private MemoryView<?> createFloatTensor(float[] data, Shape shape) {
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(DataType.FP32, shape);
        for (int i = 0; i < data.length; i++) {
            memAccess.writeFloat(memory, i * 4L, data[i]);
        }
        return MemoryViewFactory.of(DataType.FP32, memory, Layout.rowMajor(shape));
    }

    private MemoryView<?> createDoubleTensor(double[] data) {
        Shape shape = Shape.of(data.length);
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(DataType.FP64, shape);
        for (int i = 0; i < data.length; i++) {
            memAccess.writeDouble(memory, i * 8L, data[i]);
        }
        return MemoryViewFactory.of(DataType.FP64, memory, Layout.rowMajor(shape));
    }

    private MemoryView<?> createIntTensor(int[] data) {
        Shape shape = Shape.of(data.length);
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(DataType.I32, shape);
        for (int i = 0; i < data.length; i++) {
            memAccess.writeInt(memory, i * 4L, data[i]);
        }
        return MemoryViewFactory.of(DataType.I32, memory, Layout.rowMajor(shape));
    }

    private MemoryView<?> createBoolTensor(boolean[] data) {
        Shape shape = Shape.of(data.length);
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(DataType.BOOL, shape);
        for (int i = 0; i < data.length; i++) {
            memAccess.writeByte(memory, i, (byte) (data[i] ? 1 : 0));
        }
        return MemoryViewFactory.of(DataType.BOOL, memory, Layout.rowMajor(shape));
    }

    private static long floatBits(float value) {
        return Float.floatToIntBits(value);
    }

    private float readFloat(MemoryView<MemorySegment> view, long linearIndex) {
        return memAccess.readFloat(view.memory(), Indexing.linearToOffset(view, linearIndex));
    }

    private double readDouble(MemoryView<MemorySegment> view, long linearIndex) {
        return memAccess.readDouble(view.memory(), Indexing.linearToOffset(view, linearIndex));
    }

    private int readInt(MemoryView<MemorySegment> view, long linearIndex) {
        return memAccess.readInt(view.memory(), Indexing.linearToOffset(view, linearIndex));
    }

    private void assertFloatEquals(float[] expected, MemoryView<MemorySegment> actual) {
        assertEquals(expected.length, actual.shape().size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(actual, i), 0.0001f, "Mismatch at index " + i);
        }
    }

    private void assertDoubleEquals(double[] expected, MemoryView<MemorySegment> actual) {
        assertEquals(expected.length, actual.shape().size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readDouble(actual, i), 0.0001, "Mismatch at index " + i);
        }
    }

    private void assertIntEquals(int[] expected, MemoryView<MemorySegment> actual) {
        assertEquals(expected.length, actual.shape().size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readInt(actual, i), "Mismatch at index " + i);
        }
    }
}

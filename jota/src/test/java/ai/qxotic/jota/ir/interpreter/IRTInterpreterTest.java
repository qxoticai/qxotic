package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.irt.*;
import ai.qxotic.jota.ir.irt.BinaryOp;
import ai.qxotic.jota.ir.irt.UnaryOp;
import ai.qxotic.jota.memory.*;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IRTInterpreterTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUp() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void testUnaryNegate() {
        Memory<MemorySegment> inputMemory =
                context.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(3));
        MemoryAccess<MemorySegment> access = context.memoryAccess();

        access.writeFloat(inputMemory, 0, -1.0f);
        access.writeFloat(inputMemory, 1 * 4L, 2.0f);
        access.writeFloat(inputMemory, 2 * 4L, -3.0f);

        MemoryView<?> input =
                MemoryViewFactory.of(
                        DataType.FP32, inputMemory, Layout.rowMajor(Shape.of(3)));

        IRTNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        IRTNode negateOp = new UnaryOp(UnaryOperator.NEGATE, tensorInput);
        IRGraph graph = new IRGraph(List.of(tensorInput), List.of(negateOp));

        List<MemoryView<MemorySegment>> outputs =
                IRTInterpreter.execute(graph, List.of(input), context);

        assertEquals(1, outputs.size());
        MemoryView<MemorySegment> output = outputs.get(0);
        assertEquals(1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(-2.0f, readFloat(output, 1), 0.0001f);
        assertEquals(3.0f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void testBinaryAdd() {
        Memory<MemorySegment> memory1 =
                context.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(3));
        Memory<MemorySegment> memory2 =
                context.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(3));
        MemoryAccess<MemorySegment> memAccess = context.memoryAccess();

        memAccess.writeFloat(memory1, 0, 1.0f);
        memAccess.writeFloat(memory1, 1 * 4L, 2.0f);
        memAccess.writeFloat(memory1, 2 * 4L, 3.0f);
        memAccess.writeFloat(memory2, 0, 4.0f);
        memAccess.writeFloat(memory2, 1 * 4L, 5.0f);
        memAccess.writeFloat(memory2, 2 * 4L, 6.0f);

        MemoryView<?> input1 =
                MemoryViewFactory.of(DataType.FP32, memory1, Layout.rowMajor(Shape.of(3)));
        MemoryView<?> input2 =
                MemoryViewFactory.of(DataType.FP32, memory2, Layout.rowMajor(Shape.of(3)));

        IRTNode tensorInput1 = new TensorInput(0, DataType.FP32, input1.layout());
        IRTNode tensorInput2 = new TensorInput(1, DataType.FP32, input2.layout());
        IRTNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput1, tensorInput2);
        IRGraph graph = new IRGraph(List.of(tensorInput1, tensorInput2), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                IRTInterpreter.execute(graph, List.of(input1, input2), context);

        assertEquals(5.0f, readFloat(outputs.get(0), 0), 0.0001f);
        assertEquals(7.0f, readFloat(outputs.get(0), 1), 0.0001f);
        assertEquals(9.0f, readFloat(outputs.get(0), 2), 0.0001f);
    }

    @Test
    void testScalarConstantAdd() {
        Memory<MemorySegment> inputMemory =
                context.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(3));
        MemoryAccess<MemorySegment> memAccess = context.memoryAccess();

        memAccess.writeFloat(inputMemory, 0, 1.0f);
        memAccess.writeFloat(inputMemory, 1 * 4L, 2.0f);
        memAccess.writeFloat(inputMemory, 2 * 4L, 3.0f);

        MemoryView<?> input =
                MemoryViewFactory.of(DataType.FP32, inputMemory, Layout.rowMajor(Shape.of(3)));

        IRTNode tensorInput = new TensorInput(0, DataType.FP32, input.layout());
        int floatBits = Float.floatToIntBits(10.0f);
        IRTNode scalar = ScalarConstant.broadcast(floatBits, DataType.FP32, Shape.of(3));
        IRTNode addOp = new BinaryOp(BinaryOperator.ADD, tensorInput, scalar);
        IRGraph graph = new IRGraph(List.of(tensorInput), List.of(addOp));

        List<MemoryView<MemorySegment>> outputs =
                IRTInterpreter.execute(graph, List.of(input), context);

        assertEquals(11.0f, readFloat(outputs.get(0), 0), 0.0001f);
        assertEquals(12.0f, readFloat(outputs.get(0), 1), 0.0001f);
        assertEquals(13.0f, readFloat(outputs.get(0), 2), 0.0001f);
    }

    private float readFloat(MemoryView<MemorySegment> view, long linearIndex) {
        return AbstractMemoryTest.readFloat(context.memoryAccess(), view, linearIndex);
    }
}

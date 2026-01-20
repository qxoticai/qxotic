package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class CastOpsTest extends AbstractMemoryTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void castsBoolToIntegers() {
        Shape shape = Shape.of(2, 2);
        MemoryView<MemorySegment> view = boolPattern(shape, new byte[] {1, 0, 1, 0});
        Tensor input = Tensor.of(view);

        Tensor castI32 = Tracer.trace(input, t -> t.cast(DataType.I32));
        MemoryView<?> outI32 =
                ComputeEngineContext.with(new JavaComputeEngine(context), castI32::materialize);
        assertEquals(shape, outI32.shape());
        assertEquals(1, readInt(outI32, 0));
        assertEquals(0, readInt(outI32, 1));
        assertEquals(1, readInt(outI32, 2));
        assertEquals(0, readInt(outI32, 3));

        Tensor castI64 = Tracer.trace(input, t -> t.cast(DataType.I64));
        MemoryView<?> outI64 =
                ComputeEngineContext.with(new JavaComputeEngine(context), castI64::materialize);
        assertEquals(shape, outI64.shape());
        assertEquals(1L, readLong(outI64, 0));
        assertEquals(0L, readLong(outI64, 1));
        assertEquals(1L, readLong(outI64, 2));
        assertEquals(0L, readLong(outI64, 3));
    }

    @Test
    void castsIntToBool() {
        Shape shape = Shape.of(2, 2);
        MemoryView<MemorySegment> view = intPattern(shape, new int[] {0, 2, -1, 0});
        Tensor input = Tensor.of(view);
        Tensor castBool = Tracer.trace(input, t -> t.cast(DataType.BOOL));
        MemoryView<?> outBool =
                ComputeEngineContext.with(new JavaComputeEngine(context), castBool::materialize);
        assertEquals(shape, outBool.shape());
        assertEquals((byte) 0, readByte(outBool, 0));
        assertEquals((byte) 1, readByte(outBool, 1));
        assertEquals((byte) 1, readByte(outBool, 2));
        assertEquals((byte) 0, readByte(outBool, 3));
    }

    @Test
    void canary() {
        Shape shape = Shape.of(2, 2);
        MemoryView<MemorySegment> view = intPattern(shape, new int[] {0, 2, -1, 0});
        Tensor input0 = Tensor.of(view);
        Tensor input1 = Tensor.of(view);
        Tensor output =
                Tracer.trace(
                        input0,
                        input1,
                        (t0, t1) -> t0.add(t1).sum(DataType.I32).cast(DataType.FP32));
        MemoryView<?> result =
                ComputeEngineContext.with(new JavaComputeEngine(context), output::materialize);
        assertEquals(Shape.scalar(), result.shape());
        assertEquals(2.0f, readFloat(result, 0), 0.0001f);
    }

    @Test
    void castsFloatToIntLossy() {
        Shape shape = Shape.of(3);
        MemoryView<MemorySegment> view = floatPattern(shape, new float[] {1.8f, -2.4f, 0.0f});
        Tensor input = Tensor.of(view);

        Tensor castI32 = Tracer.trace(input, t -> t.cast(DataType.I32));
        MemoryView<?> outI32 =
                ComputeEngineContext.with(new JavaComputeEngine(context), castI32::materialize);
        assertEquals(shape, outI32.shape());
        assertEquals(1, readInt(outI32, 0));
        assertEquals(-2, readInt(outI32, 1));
        assertEquals(0, readInt(outI32, 2));
    }

    private MemoryView<MemorySegment> boolPattern(Shape shape, byte[] values) {
        MemoryView<MemorySegment> view =
                MemoryHelpers.full(context, DataType.BOOL, shape.size(), 0).view(shape);
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = view.byteOffset() + (long) i * DataType.BOOL.byteSize();
            access.writeByte(view.memory(), offset, values[i]);
        }
        return view;
    }

    private MemoryView<MemorySegment> intPattern(Shape shape, int[] values) {
        MemoryView<MemorySegment> view =
                MemoryHelpers.full(context, DataType.I32, shape.size(), 0).view(shape);
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = view.byteOffset() + (long) i * DataType.I32.byteSize();
            access.writeInt(view.memory(), offset, values[i]);
        }
        return view;
    }

    private MemoryView<MemorySegment> floatPattern(Shape shape, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryHelpers.full(context, DataType.FP32, shape.size(), 0f).view(shape);
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = view.byteOffset() + (long) i * DataType.FP32.byteSize();
            access.writeFloat(view.memory(), offset, values[i]);
        }
        return view;
    }

    private byte readByte(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_BYTE, offset);
    }

    private int readInt(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }

    private long readLong(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
    }

    private float readFloat(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }
}

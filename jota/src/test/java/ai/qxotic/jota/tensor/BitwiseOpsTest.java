package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class BitwiseOpsTest {

    private static final List<DataType> INTEGRAL_TYPES =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void bitwiseNotWorksForIntegralTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : INTEGRAL_TYPES) {
            MemoryView<MemorySegment> left =
                    MemoryHelpers.arange(context, dataType, shape.size()).view(shape);
            Tensor input = Tensor.of(left);
            Tensor output = Tracer.trace(input, Tensor::bitwiseNot);
            MemoryView<?> result = output.materialize();

            for (int i = 0; i < shape.size(); i++) {
                long value = readIntegral(result, i, dataType);
                long expected = castIntegral(~i, dataType);
                assertEquals(expected, value, "Mismatch for " + dataType + " at " + i);
            }
        }
    }

    @Test
    void bitwiseBinaryOpsWorkForIntegralTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : INTEGRAL_TYPES) {
            MemoryView<MemorySegment> left =
                    MemoryHelpers.arange(context, dataType, shape.size()).view(shape);
            MemoryView<MemorySegment> right =
                    MemoryHelpers.full(context, dataType, shape.size(), 3).view(shape);
            Tensor a = Tensor.of(left);
            Tensor b = Tensor.of(right);

            Tensor andTensor = Tracer.trace(a, b, Tensor::bitwiseAnd);
            Tensor orTensor = Tracer.trace(a, b, Tensor::bitwiseOr);
            Tensor xorTensor = Tracer.trace(a, b, Tensor::bitwiseXor);

            MemoryView<?> andOut = andTensor.materialize();
            MemoryView<?> orOut = orTensor.materialize();
            MemoryView<?> xorOut = xorTensor.materialize();

            for (int i = 0; i < shape.size(); i++) {
                long leftValue = castIntegral(i, dataType);
                long rightValue = castIntegral(3, dataType);
                assertEquals(
                        castIntegral((int) leftValue & (int) rightValue, dataType),
                        readIntegral(andOut, i, dataType),
                        "bitwiseAnd mismatch for " + dataType + " at " + i);
                assertEquals(
                        castIntegral((int) leftValue | (int) rightValue, dataType),
                        readIntegral(orOut, i, dataType),
                        "bitwiseOr mismatch for " + dataType + " at " + i);
                assertEquals(
                        castIntegral((int) leftValue ^ (int) rightValue, dataType),
                        readIntegral(xorOut, i, dataType),
                        "bitwiseXor mismatch for " + dataType + " at " + i);
            }
        }
    }

    @Test
    void bitwiseOpsRejectBoolAndFloat() {
        Shape shape = Shape.of(2);
        MemoryView<MemorySegment> boolView =
                MemoryHelpers.full(context, DataType.BOOL, shape.size(), 1).view(shape);
        Tensor boolTensor = Tensor.of(boolView);
        assertThrows(
                IllegalArgumentException.class, () -> Tracer.trace(boolTensor, Tensor::bitwiseNot));

        MemoryView<MemorySegment> floatView =
                MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);
        Tensor floatTensor = Tensor.of(floatView);
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(floatTensor, Tensor::bitwiseNot));
    }

    private long readIntegral(MemoryView<?> view, long index, DataType dataType) {
        long offset = Indexing.linearToOffset(view, index);
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> castView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        if (dataType == DataType.I8) {
            return access.readByte(castView.memory(), offset);
        }
        if (dataType == DataType.I16) {
            return access.readShort(castView.memory(), offset);
        }
        if (dataType == DataType.I32) {
            return access.readInt(castView.memory(), offset);
        }
        if (dataType == DataType.I64) {
            return access.readLong(castView.memory(), offset);
        }
        throw new IllegalStateException("Unsupported integral type: " + dataType);
    }

    private long castIntegral(long value, DataType dataType) {
        if (dataType == DataType.I8) {
            return (byte) value;
        }
        if (dataType == DataType.I16) {
            return (short) value;
        }
        if (dataType == DataType.I32) {
            return (int) value;
        }
        if (dataType == DataType.I64) {
            return value;
        }
        throw new IllegalStateException("Unsupported integral type: " + dataType);
    }
}

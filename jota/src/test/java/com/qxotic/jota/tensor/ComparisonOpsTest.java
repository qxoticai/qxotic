package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ComparisonOpsTest {

    private static final List<DataType> PRIMITIVE_DATA_TYPES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void derivedComparisonsWorkAcrossTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            MemoryView<MemorySegment> left = range(dataType, shape);
            MemoryView<MemorySegment> right = fullZeros(dataType, shape);
            Tensor leftTensor = Tensor.of(left);
            Tensor rightTensor = Tensor.of(right);

            Tensor notEqual = Tracer.trace(leftTensor, rightTensor, Tensor::notEqual);
            Tensor greaterThan = Tracer.trace(leftTensor, rightTensor, Tensor::greaterThan);
            Tensor lessThanOrEqual = Tracer.trace(leftTensor, rightTensor, Tensor::lessThanOrEqual);
            Tensor greaterThanOrEqual =
                    Tracer.trace(leftTensor, rightTensor, Tensor::greaterThanOrEqual);

            MemoryView<?> notEqualOut = notEqual.materialize();
            MemoryView<?> greaterThanOut = greaterThan.materialize();
            MemoryView<?> lessThanOrEqualOut = lessThanOrEqual.materialize();
            MemoryView<?> greaterThanOrEqualOut = greaterThanOrEqual.materialize();

            for (int i = 0; i < shape.size(); i++) {
                int leftValue = dataType == DataType.BOOL ? (i % 2) : i;
                int rightValue = 0;
                assertEquals(leftValue != rightValue ? 1 : 0, readBool(notEqualOut, i));
                assertEquals(leftValue > rightValue ? 1 : 0, readBool(greaterThanOut, i));
                assertEquals(leftValue <= rightValue ? 1 : 0, readBool(lessThanOrEqualOut, i));
                assertEquals(leftValue >= rightValue ? 1 : 0, readBool(greaterThanOrEqualOut, i));
            }

            Tensor flippedGreaterThan = Tracer.trace(rightTensor, leftTensor, Tensor::greaterThan);
            MemoryView<?> flippedGreaterOut = flippedGreaterThan.materialize();
            for (int i = 0; i < shape.size(); i++) {
                int leftValue = dataType == DataType.BOOL ? (i % 2) : i;
                int rightValue = 0;
                assertEquals(rightValue > leftValue ? 1 : 0, readBool(flippedGreaterOut, i));
            }
        }
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return boolPattern(shape);
        }
        return MemoryHelpers.arange(domain, dataType, shape.size()).view(shape);
    }

    private MemoryView<MemorySegment> fullZeros(DataType dataType, Shape shape) {
        return MemoryHelpers.full(domain, dataType, shape.size(), 0).view(shape);
    }

    private MemoryView<MemorySegment> boolPattern(Shape shape) {
        MemoryView<MemorySegment> view =
                MemoryHelpers.full(domain, DataType.BOOL, shape.size(), 0).view(shape);
        for (int i = 0; i < shape.size(); i++) {
            long offset = view.byteOffset() + (long) i * DataType.BOOL.byteSize();
            byte value = (byte) (i % 2);
            domain.directAccess().writeByte(view.memory(), offset, value);
        }
        return view;
    }

    private byte readBool(MemoryView<?> view, long index) {
        long offset = Indexing.linearToOffset(view, index);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_BYTE, offset);
    }
}

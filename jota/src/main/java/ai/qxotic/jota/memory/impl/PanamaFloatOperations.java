package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.DataType;
import ai.llm4j.jota.FloatBinaryOperator;
import ai.llm4j.jota.FloatUnaryOperator;
import ai.llm4j.jota.memory.OffsetIterator;
import ai.qxotic.jota.memory.FloatOperations;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;

import java.lang.foreign.MemorySegment;
import java.util.NoSuchElementException;
import java.util.Objects;

final class PanamaFloatOperations implements FloatOperations<MemorySegment> {

    final MemoryAccess<MemorySegment> memoryAccess;

    PanamaFloatOperations(MemoryAccess<MemorySegment> memoryAccess) {
        this.memoryAccess = Objects.requireNonNull(memoryAccess);
    }

    @Override
    public void elementWise(MemoryView<MemorySegment> in, FloatUnaryOperator unaryOperator, MemoryView<MemorySegment> out) {
        if (in.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (in) data type: " + in.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(in.shape(), out.shape())) {
            throw new IllegalArgumentException("Incompatible input/output shapes, in: " + in.shape() + " out: " + out.shape());
        }
        var inIterator = OffsetIterator.create(in);
        var outIterator = OffsetIterator.create(out);
        while (inIterator.hasNext() && outIterator.hasNext()) {
            long inByteOffset = inIterator.nextByteOffset();
            long outByteOffset = outIterator.nextByteOffset();
            float inValue = memoryAccess.readFloat(in.memory(), inByteOffset);
            memoryAccess.writeFloat(out.memory(), outByteOffset, unaryOperator.applyAsFloat(inValue));
        }
    }

    @Override
    public void elementWise(float scalar, FloatUnaryOperator unaryOperator, MemoryView<MemorySegment> out) {
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        var outIterator = OffsetIterator.create(out);
        while (outIterator.hasNext()) {
            long outByteOffset = outIterator.nextByteOffset();
            memoryAccess.writeFloat(out.memory(), outByteOffset, unaryOperator.applyAsFloat(scalar));
        }
    }

    @Override
    public void elementWise2(MemoryView<MemorySegment> left, FloatBinaryOperator binaryOperator, MemoryView<MemorySegment> right, MemoryView<MemorySegment> out) {
        if (left.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (left) data type: " + left.dataType());
        }
        if (right.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (right) data type: " + right.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(left.shape(), right.shape())) {
            throw new IllegalArgumentException("Incompatible input shapes, left: " + left.shape() + " right: " + right.shape());
        }
        if (!Objects.equals(left.shape(), out.shape())) {
            throw new IllegalArgumentException("Incompatible output shape, expected: " + left.shape() + " but got: " + out.shape());
        }
        var leftIterator = OffsetIterator.create(left);
        var rightIterator = OffsetIterator.create(right);
        var outIterator = OffsetIterator.create(out);
        while (leftIterator.hasNext() && rightIterator.hasNext() && outIterator.hasNext()) {
            long leftByteOffset = leftIterator.nextByteOffset();
            long rightByteOffset = rightIterator.nextByteOffset();
            long outByteOffset = outIterator.nextByteOffset();
            float leftValue = memoryAccess.readFloat(left.memory(), leftByteOffset);
            float rightValue = memoryAccess.readFloat(right.memory(), rightByteOffset);
            memoryAccess.writeFloat(out.memory(), outByteOffset, binaryOperator.applyAsFloat(leftValue, rightValue));
        }
    }

    @Override
    public void elementWise2(MemoryView<MemorySegment> left, FloatBinaryOperator binaryOperator, float right, MemoryView<MemorySegment> out) {
        if (left.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (left) data type: " + left.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(left.shape(), out.shape())) {
            throw new IllegalArgumentException("Incompatible output shape, expected: " + left.shape() + " but got: " + out.shape());
        }
        var leftIterator = OffsetIterator.create(left);
        var outIterator = OffsetIterator.create(out);
        while (leftIterator.hasNext() && outIterator.hasNext()) {
            long leftByteOffset = leftIterator.nextByteOffset();
            long outByteOffset = outIterator.nextByteOffset();
            float leftValue = memoryAccess.readFloat(left.memory(), leftByteOffset);
            memoryAccess.writeFloat(out.memory(), outByteOffset, binaryOperator.applyAsFloat(leftValue, right));
        }
    }

    @Override
    public void fold(MemoryView<MemorySegment> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<MemorySegment> out, int _axis) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void reduce(MemoryView<MemorySegment> in, FloatBinaryOperator binaryOperator, MemoryView<MemorySegment> out, int _axis) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float reduceAll(MemoryView<MemorySegment> in, FloatBinaryOperator binaryOperator) {
        if (in.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported data type: " + in.dataType());
        }
        var iterator = OffsetIterator.create(in);
        if (!iterator.hasNext()) {
            throw new NoSuchElementException();
        }
        float accumulator = iterator.nextByteOffset();
        while (iterator.hasNext()) {
            long byteOffset = iterator.nextByteOffset();
            float value = memoryAccess.readFloat(in.memory(), byteOffset);
            accumulator = binaryOperator.applyAsFloat(accumulator, value);
        }
        return accumulator;
    }

    @Override
    public float foldAll(MemoryView<MemorySegment> in, float initialValue, FloatBinaryOperator binaryOperator) {
        if (in.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported data type: " + in.dataType());
        }
        float accumulator = initialValue;
        for (var iterator = OffsetIterator.create(in); iterator.hasNext(); ) {
            long byteOffset = iterator.nextByteOffset();
            float value = memoryAccess.readFloat(in.memory(), byteOffset);
            accumulator = binaryOperator.applyAsFloat(accumulator, value);
        }
        return accumulator;
    }

    @Override
    public void matrixMultiply(MemoryView<MemorySegment> left, MemoryView<MemorySegment> right, MemoryView<MemorySegment> out) {
        throw new UnsupportedOperationException();
    }
}

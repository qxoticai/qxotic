package com.llm4j.jota.memory.impl;

import com.llm4j.jota.DataType;
import com.llm4j.jota.FloatBinaryOperator;
import com.llm4j.jota.FloatUnaryOperator;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.OffsetIterator;
import com.llm4j.jota.memory.FloatOperations;
import com.llm4j.jota.memory.MemoryAccess;
import com.llm4j.jota.memory.MemoryView;

import java.nio.ByteBuffer;
import java.util.NoSuchElementException;
import java.util.Objects;

final class ByteBufferFloatOperations implements FloatOperations<ByteBuffer> {

    final MemoryAccess<ByteBuffer> memoryAccess;

    ByteBufferFloatOperations(MemoryAccess<ByteBuffer> memoryAccess) {
        this.memoryAccess = Objects.requireNonNull(memoryAccess);
    }

    @Override
    public void elementWise(MemoryView<ByteBuffer> in, FloatUnaryOperator unaryOperator, MemoryView<ByteBuffer> out) {
        if (in.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (in) data type: " + in.dataType());
        }
        if (out.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Shape.sameAs(in.shape(), out.shape())) {
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
    public void elementWise(float scalar, FloatUnaryOperator unaryOperator, MemoryView<ByteBuffer> out) {
        if (out.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        var outIterator = OffsetIterator.create(out);
        while (outIterator.hasNext()) {
            long outByteOffset = outIterator.nextByteOffset();
            memoryAccess.writeFloat(out.memory(), outByteOffset, unaryOperator.applyAsFloat(scalar));
        }
    }

    @Override
    public void elementWise2(MemoryView<ByteBuffer> left, FloatBinaryOperator binaryOperator, MemoryView<ByteBuffer> right, MemoryView<ByteBuffer> out) {
        if (left.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (left) data type: " + left.dataType());
        }
        if (right.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (right) data type: " + right.dataType());
        }
        if (out.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Shape.sameAs(left.shape(), right.shape())) {
            throw new IllegalArgumentException("Incompatible input shapes, left: " + left.shape() + " right: " + right.shape());
        }
        if (!Shape.sameAs(left.shape(), out.shape())) {
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
    public void elementWise2(MemoryView<ByteBuffer> left, FloatBinaryOperator binaryOperator, float right, MemoryView<ByteBuffer> out) {
        if (left.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (left) data type: " + left.dataType());
        }
        if (out.dataType() != DataType.F32) {
            throw new UnsupportedOperationException("Unsupported (out) data type: " + out.dataType());
        }
        if (!Shape.sameAs(left.shape(), out.shape())) {
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
    public void fold(MemoryView<ByteBuffer> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<ByteBuffer> out, int _axis) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void reduce(MemoryView<ByteBuffer> in, FloatBinaryOperator binaryOperator, MemoryView<ByteBuffer> out, int _axis) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float reduceAll(MemoryView<ByteBuffer> in, FloatBinaryOperator binaryOperator) {
        if (in.dataType() != DataType.F32) {
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
    public float foldAll(MemoryView<ByteBuffer> in, float initialValue, FloatBinaryOperator binaryOperator) {
        if (in.dataType() != DataType.F32) {
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
    public void matrixMultiply(MemoryView<ByteBuffer> left, MemoryView<ByteBuffer> right, MemoryView<ByteBuffer> out) {
        throw new UnsupportedOperationException();
    }
}

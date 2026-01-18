package ai.qxotic.jota.memory.impl;

import ai.llm4j.jota.FloatBinaryOperator;
import ai.llm4j.jota.FloatUnaryOperator;
import ai.llm4j.jota.memory.OffsetIterator;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.FloatOperations;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import java.util.NoSuchElementException;
import java.util.Objects;

final class GenericFloatOperations<B> implements FloatOperations<B> {

    final MemoryAccess<B> memoryAccess;

    GenericFloatOperations(MemoryAccess<B> memoryAccess) {
        this.memoryAccess = Objects.requireNonNull(memoryAccess);
    }

    @Override
    public void elementWise(MemoryView<B> in, FloatUnaryOperator unaryOperator, MemoryView<B> out) {
        if (in.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported (in) data type: " + in.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(in.shape(), out.shape())) {
            throw new IllegalArgumentException(
                    "Incompatible input/output shapes, in: " + in.shape() + " out: " + out.shape());
        }
        var inIterator = OffsetIterator.create(in);
        var outIterator = OffsetIterator.create(out);
        while (inIterator.hasNext() && outIterator.hasNext()) {
            long inByteOffset = inIterator.nextByteOffset();
            long outByteOffset = outIterator.nextByteOffset();
            float inValue = memoryAccess.readFloat(in.memory(), inByteOffset);
            memoryAccess.writeFloat(
                    out.memory(), outByteOffset, unaryOperator.applyAsFloat(inValue));
        }
    }

    @Override
    public void elementWise(float scalar, FloatUnaryOperator unaryOperator, MemoryView<B> out) {
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (out) data type: " + out.dataType());
        }
        var outIterator = OffsetIterator.create(out);
        while (outIterator.hasNext()) {
            long outByteOffset = outIterator.nextByteOffset();
            memoryAccess.writeFloat(
                    out.memory(), outByteOffset, unaryOperator.applyAsFloat(scalar));
        }
    }

    @Override
    public void elementWise2(
            MemoryView<B> left,
            FloatBinaryOperator binaryOperator,
            MemoryView<B> right,
            MemoryView<B> out) {
        if (left.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (left) data type: " + left.dataType());
        }
        if (right.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (right) data type: " + right.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(left.shape(), right.shape())) {
            throw new IllegalArgumentException(
                    "Incompatible input shapes, left: "
                            + left.shape()
                            + " right: "
                            + right.shape());
        }
        if (!Objects.equals(left.shape(), out.shape())) {
            throw new IllegalArgumentException(
                    "Incompatible output shape, expected: "
                            + left.shape()
                            + " but got: "
                            + out.shape());
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
            memoryAccess.writeFloat(
                    out.memory(),
                    outByteOffset,
                    binaryOperator.applyAsFloat(leftValue, rightValue));
        }
    }

    @Override
    public void elementWise2(
            MemoryView<B> left,
            FloatBinaryOperator binaryOperator,
            float right,
            MemoryView<B> out) {
        if (left.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (left) data type: " + left.dataType());
        }
        if (out.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException(
                    "Unsupported (out) data type: " + out.dataType());
        }
        if (!Objects.equals(left.shape(), out.shape())) {
            throw new IllegalArgumentException(
                    "Incompatible output shape, expected: "
                            + left.shape()
                            + " but got: "
                            + out.shape());
        }
        var leftIterator = OffsetIterator.create(left);
        var outIterator = OffsetIterator.create(out);
        while (leftIterator.hasNext() && outIterator.hasNext()) {
            long leftByteOffset = leftIterator.nextByteOffset();
            long outByteOffset = outIterator.nextByteOffset();
            float leftValue = memoryAccess.readFloat(left.memory(), leftByteOffset);
            memoryAccess.writeFloat(
                    out.memory(), outByteOffset, binaryOperator.applyAsFloat(leftValue, right));
        }
    }

    @Override
    public void fold(
            MemoryView<B> in,
            FloatBinaryOperator binaryOperator,
            float initialValue,
            MemoryView<B> out,
            int _axis) {
        throw new UnsupportedOperationException();
        //        if (in.dataType() != DataType.FP32) {
        //            throw new UnsupportedOperationException("Unsupported (in) data type: " +
        // in.dataType());
        //        }
        //        if (out.dataType() != DataType.FP32) {
        //            throw new UnsupportedOperationException("Unsupported (out) data type: " +
        // out.dataType());
        //        }
        //
        //        int axis = ai.qxotic.jota.Util.wrapAround(_axis, in.shape().rank());
        //
        //        // Output shape should have the axis dimension replaced with size 1
        //        ai.qxotic.jota.Shape expectedShape = in.shape().replace(axis,
        // ai.qxotic.jota.Shape.flat(1));
        //        if (!expectedShape.isCongruentWith(out.shape())) {
        //            throw new IllegalArgumentException("Incompatible shape, expected " +
        // expectedShape + " but got " + out.shape());
        //        }
        //
        //        long axisSize = in.shape().size(axis);
        //        if (axisSize == 0) {
        //            throw new IllegalArgumentException("Cannot fold along axis with size 0");
        //        }
        //
        //        // Iterate through output elements
        //        var outIterator = OffsetIterator.create(out);
        //        while (outIterator.hasNext()) {
        //            long outByteOffset = outIterator.nextByteOffset();
        //
        //            // For each output position, we need to fold along the axis
        //            // Get the corresponding input slice
        //            long[] outCoords = outIterator.currentCoordinates();
        //
        //            // Perform fold along axis
        //            float accumulator = initialValue;
        //            for (long i = 0; i < axisSize; i++) {
        //                // Build input coordinates by setting axis coordinate to i
        //                long[] inCoords = outCoords.clone();
        //                inCoords[axis] = i;
        //
        //                long inByteOffset = calculateByteOffset(in, inCoords);
        //                float value = memoryAccess.readFloat(in.memory(), inByteOffset);
        //                accumulator = binaryOperator.applyAsFloat(accumulator, value);
        //            }
        //
        //            memoryAccess.writeFloat(out.memory(), outByteOffset, accumulator);
        //        }
    }

    private long calculateByteOffset(MemoryView<B> view, long[] coordinates) {
        long offset = view.byteOffset();
        long[] strides = view.byteStride().toArray();
        for (int i = 0; i < coordinates.length; i++) {
            offset += coordinates[i] * strides[i];
        }
        return offset;
    }

    //        Shape expectedShape = in.shape().remove(_axis);
    //        if (!Objects.equals(expectedShape, out.shape())) {
    //            throw new IllegalArgumentException("Incompatible shape, expected " + expectedShape
    // + " but got " + out.shape());
    //        }
    //
    //        Shape inShape = in.shape();
    //        Shape outShape = out.shape();
    //
    //        // Validate that output dimensions match input dimensions (excluding the reduced axis)
    //        int outDimIndex = 0;
    //        for (int inDimIndex = 0; inDimIndex < inShape.rank(); inDimIndex++) {
    //            if (inDimIndex == _axis) {
    //                continue; // Skip the reduced dimension
    //            }
    //            if (outDimIndex >= outShape.rank() ||
    //                    inShape.dimension(inDimIndex) != outShape.dimension(outDimIndex)) {
    //                throw new IllegalArgumentException("Output shape doesn't match expected
    // reduced shape");
    //            }
    //            outDimIndex++;
    //        }
    //
    //        long[] inStrides = in.byteStride();
    //        long axisSize = inShape.dimension(_axis);
    //        long axisStrideBytes = inStrides[_axis];
    //
    //        // Calculate total number of reduction operations needed
    //        long totalOutputElements = outShape.size();
    //
    //        // Iterate through each output position
    //        long[] outCoordinates = new long[outShape.rank()];
    //
    //        for (long outIndex = 0; outIndex < totalOutputElements; outIndex++) {
    //            // Convert linear output index to coordinates
    //            long tempIndex = outIndex;
    //            for (int dim = outShape.rank() - 1; dim >= 0; dim--) {
    //                long dimSize = outShape.dimension(dim);
    //                outCoordinates[dim] = tempIndex % dimSize;
    //                tempIndex /= dimSize;
    //            }
    //
    //            // Map output coordinates to input coordinates (inserting the axis dimension)
    //            long[] inCoordinates = new long[inShape.rank()];
    //            int inDimIndex = 0;
    //            for (outDimIndex = 0; outDimIndex < outShape.rank(); outDimIndex++) {
    //                if (inDimIndex == _axis) {
    //                    inDimIndex++; // Skip the axis dimension in input coordinates
    //                }
    //                inCoordinates[inDimIndex] = outCoordinates[outDimIndex];
    //                inDimIndex++;
    //            }
    //
    //            // Calculate base input offset (without the axis coordinate)
    //            long baseInOffset = in.byteOffset();
    //            for (int dim = 0; dim < inShape.rank(); dim++) {
    //                if (dim != _axis) {
    //                    baseInOffset += inCoordinates[dim] * inStrides[dim];
    //                }
    //            }
    //
    //            // Perform fold operation along the axis
    //            float accumulator = initialValue;
    //            for (long i = 0; i < axisSize; i++) {
    //                long inOffset = baseInOffset + i * axisStrideBytes;
    //                float inputValue = memoryAccess.readFloat(in.memory(), inOffset);
    //                accumulator = binaryOperator.applyAsFloat(accumulator, inputValue);
    //            }
    //
    //            // Calculate output offset and write result
    //            long outOffset = out.byteOffset();
    //            long[] outStrides = out.byteStride();
    //            for (int dim = 0; dim < outShape.rank(); dim++) {
    //                outOffset += outCoordinates[dim] * outStrides[dim];
    //            }
    //
    //            memoryAccess.writeFloat(out.memory(), outOffset, accumulator);
    //        }
    //    }

    @Override
    public void reduce(
            MemoryView<B> in, FloatBinaryOperator binaryOperator, MemoryView<B> out, int _axis) {
        throw new UnsupportedOperationException();
        //        Shape expectedShape = in.shape().remove(_axis);
        //        if (!Objects.equals(expectedShape, out.shape())) {
        //            throw new IllegalArgumentException("Incompatible shape, expected " +
        // expectedShape + " but got " + out.shape());
        //        }
        //
        //        Shape inShape = in.shape();
        //        Shape outShape = out.shape();
        //
        //        // Validate axis
        //        if (_axis < 0 || _axis >= inShape.rank()) {
        //            throw new IllegalArgumentException("Axis " + _axis + " is out of bounds for
        // tensor with " + inShape.rank() + " dimensions");
        //        }
        //
        //        // Validate that the axis has at least one element to reduce
        //        long axisSize = inShape.dimension(_axis);
        //        if (axisSize == 0) {
        //            throw new IllegalArgumentException("Cannot reduce along axis with size 0");
        //        }
        //
        //        // Validate output shape - should have one less dimension than input
        //        if (outShape.rank() != inShape.rank() - 1) {
        //            throw new IllegalArgumentException("Output rank should be " + (inShape.rank()
        // - 1) + " but got " + outShape.rank());
        //        }
        //
        //        // Validate that output dimensions match input dimensions (excluding the reduced
        // axis)
        //        int outDimIndex = 0;
        //        for (int inDimIndex = 0; inDimIndex < inShape.rank(); inDimIndex++) {
        //            if (inDimIndex == _axis) {
        //                continue; // Skip the reduced dimension
        //            }
        //            if (outDimIndex >= outShape.rank() ||
        //                    inShape.dimension(inDimIndex) != outShape.dimension(outDimIndex)) {
        //                throw new IllegalArgumentException("Output shape doesn't match expected
        // reduced shape");
        //            }
        //            outDimIndex++;
        //        }
        //
        //        long[] inStrides = in.byteStride();
        //        long axisStrideBytes = inStrides[_axis];
        //
        //        // Calculate total number of reduction operations needed
        //        long totalOutputElements = outShape.size();
        //
        //        // Iterate through each output position
        //        long[] outCoordinates = new long[outShape.rank()];
        //
        //        for (long outIndex = 0; outIndex < totalOutputElements; outIndex++) {
        //            // Convert linear output index to coordinates
        //            long tempIndex = outIndex;
        //            for (int dim = outShape.rank() - 1; dim >= 0; dim--) {
        //                long dimSize = outShape.dimension(dim);
        //                outCoordinates[dim] = tempIndex % dimSize;
        //                tempIndex /= dimSize;
        //            }
        //
        //            // Map output coordinates to input coordinates (inserting the axis dimension)
        //            long[] inCoordinates = new long[inShape.rank()];
        //            int inDimIndex = 0;
        //            for (outDimIndex = 0; outDimIndex < outShape.rank(); outDimIndex++) {
        //                if (inDimIndex == _axis) {
        //                    inDimIndex++; // Skip the axis dimension in input coordinates
        //                }
        //                inCoordinates[inDimIndex] = outCoordinates[outDimIndex];
        //                inDimIndex++;
        //            }
        //
        //            // Calculate base input offset (without the axis coordinate)
        //            long baseInOffset = in.byteOffset();
        //            for (int dim = 0; dim < inShape.rank(); dim++) {
        //                if (dim != _axis) {
        //                    baseInOffset += inCoordinates[dim] * inStrides[dim];
        //                }
        //            }
        //
        //            // Perform reduce operation along the axis
        //            // Start with the first element (no initial value unlike fold)
        //            long firstInOffset = baseInOffset;
        //            float accumulator = memoryAccess.readFloat(in.memory(), firstInOffset);
        //
        //            // Continue with remaining elements
        //            for (long i = 1; i < axisSize; i++) {
        //                long inOffset = baseInOffset + i * axisStrideBytes;
        //                float inputValue = memoryAccess.readFloat(in.memory(), inOffset);
        //                accumulator = binaryOperator.applyAsFloat(accumulator, inputValue);
        //            }
        //
        //            // Calculate output offset and write result
        //            long outOffset = out.byteOffset();
        //            long[] outStrides = out.byteStride();
        //            for (int dim = 0; dim < outShape.rank(); dim++) {
        //                outOffset += outCoordinates[dim] * outStrides[dim];
        //            }
        //
        //            memoryAccess.writeFloat(out.memory(), outOffset, accumulator);
        //        }
    }

    @Override
    public float reduceAll(MemoryView<B> in, FloatBinaryOperator binaryOperator) {
        if (in.dataType() != DataType.FP32) {
            throw new UnsupportedOperationException("Unsupported data type: " + in.dataType());
        }
        var iterator = OffsetIterator.create(in);
        if (!iterator.hasNext()) {
            throw new NoSuchElementException();
        }
        float accumulator = memoryAccess.readFloat(in.memory(), iterator.nextByteOffset());
        while (iterator.hasNext()) {
            long byteOffset = iterator.nextByteOffset();
            float value = memoryAccess.readFloat(in.memory(), byteOffset);
            accumulator = binaryOperator.applyAsFloat(accumulator, value);
        }
        return accumulator;
    }

    @Override
    public float foldAll(MemoryView<B> in, float initialValue, FloatBinaryOperator binaryOperator) {
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
    public void matrixMultiply(MemoryView<B> left, MemoryView<B> right, MemoryView<B> out) {
        throw new UnsupportedOperationException();
    }
}

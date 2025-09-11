package com.llm4j.jota.memory;

import com.llm4j.jota.FloatBinaryOperator;
import com.llm4j.jota.FloatUnaryOperator;
import com.llm4j.jota.Shape;

import static com.llm4j.jota.FloatBinaryOperator.*;
import static com.llm4j.jota.FloatUnaryOperator.exp;
import static com.llm4j.jota.FloatUnaryOperator.square;

public interface FloatOperations<B> {

    void elementWise(MemoryView<B> in, FloatUnaryOperator unaryOperator, MemoryView<B> out);

    // Scalar version.
    void elementWise(float in, FloatUnaryOperator unaryOperator, MemoryView<B> out);

    default float elementWise(float in, FloatUnaryOperator unaryOperator) {
        return unaryOperator.applyAsFloat(in);
    }

    void elementWise2(MemoryView<B> left, FloatBinaryOperator binaryOperator, MemoryView<B> right, MemoryView<B> out);

    // Scalar version.
    void elementWise2(MemoryView<B> left, FloatBinaryOperator binaryOperator, float right, MemoryView<B> out);

    void fold(MemoryView<B> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<B> out, int _axis);

    void reduce(MemoryView<B> in, FloatBinaryOperator binaryOperator, MemoryView<B> out, int _axis);

    float reduceAll(MemoryView<B> in, FloatBinaryOperator binaryOperator);

    float foldAll(MemoryView<B> in, float initialValue, FloatBinaryOperator binaryOperator);

    default float minAll(MemoryView<B> in) {
        return foldAll(in, Float.POSITIVE_INFINITY, min());
    }

    default float maxAll(MemoryView<B> in) {
        return foldAll(in, Float.NEGATIVE_INFINITY, max());
    }

    default float sumAll(MemoryView<B> in) {
        return foldAll(in, 0f, sum());
    }

    default float multiplyAll(MemoryView<B> in) {
        return foldAll(in, 1f, product());
    }

    default float meanAll(MemoryView<B> in) {
        return sumAll(in) / in.shape().totalNumberOfElements();
    }

    // Multiply (R, C) @ (C, 1) = (R)
    // default void matrixVectorMultiply(MemoryView<B> matrix, MemoryView<B> vector, MemoryView<B> out);

    // Multiply (R, K) @ (K, C) = (R, C)
    void matrixMultiply(MemoryView<B> left, MemoryView<B> right, MemoryView<B> out);

//
//    void gemmRowMajor(long R, long C, long K,
//                      FloatSpan a, long aOffset, long aRowStride, // [R, K]
//                      FloatSpan b, long bOffset, long bRowStride, // [K, C]^T
//                      FloatSpan out, long outOffset, long outRowStride); // [R, C]

    //void rotate(boolean neoxStyle, V span, V freqReal, V freqImag, int position, int numberOfHeads, int headSize, V out);

    default void assign(float scalar, MemoryView<B> out) {
        elementWise(scalar, FloatUnaryOperator.identity(), out);
    }

    default void assign(MemoryView<B> in, MemoryView<B> out) {
        elementWise(in, FloatUnaryOperator.identity(), out);
    }

    default void add(MemoryView<B> left, MemoryView<B> right, MemoryView<B> out) {
        elementWise2(left, sum(), right, out);
    }

    default void multiply(MemoryView<B> left, MemoryView<B> right, MemoryView<B> out) {
        elementWise2(left, product(), right, out);
    }

    default void multiply(MemoryView<B> left, float right, MemoryView<B> out) {
        elementWise2(left, product(), right, out);
    }

    default void softMax(MemoryView<B> in, MemoryView<B> out, MemoryView<B> tmp, int axis) {
        // 1. Validate shapes
        if (!Shape.sameAs(in.shape(), out.shape())) {
            throw new IllegalArgumentException("Input and output shapes must match");
        }
        // 2. Create temporary reduced views
        Shape reducedShape = in.shape().replace(axis, 1);
        if (!Shape.sameAs(reducedShape, tmp.shape())) {
            throw new IllegalArgumentException("Invalid tmp shape");
        }
        if (axis < 0 || axis >= in.shape().rank()) {
            throw new IllegalArgumentException("Invalid axis: " + axis);
        }

        // 3. Compute max along axis (for numerical stability)
        fold(in, FloatBinaryOperator.max(), Float.NEGATIVE_INFINITY, tmp, axis);

        // 4. Compute exp(x - max)
        elementWise2(in, subtract().andThen(exp()), tmp, out);

        // 5. Sum exponentials along axis
        fold(out, sum(), 0f, tmp, axis);

        // 6. Normalize
        elementWise2(out, divide(), tmp.expand(out.shape()), out);
    }

    default void softMax(MemoryView<B> in, MemoryView<B> out) {
        // find max value (for numerical stability)
        float maxValue = maxAll(in);
        // exp and sum
        elementWise2(in, subtract().andThen(exp()), maxValue, out); // exp(f - maxValue)
        float sum = sumAll(out);
        // normalize
        elementWise2(out, FloatBinaryOperator.divide(), sum, out);
    }


    default void rootMeanSquareNorm(MemoryView<B> in, MemoryView<B> weight, float rmsNormEps, MemoryView<B> out) {
        // calculate sum of squares
        float ss = foldAll(in, 0f, sum().accumulate(square()));
        ss /= in.shape().totalNumberOfElements();
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        elementWise2(in, product(), weight, out); // out = in * weight
        elementWise2(out, product(), ss, out);      // out = out * ss
    }
}

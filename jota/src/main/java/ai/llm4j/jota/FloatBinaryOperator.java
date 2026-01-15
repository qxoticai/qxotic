package ai.llm4j.jota;

import java.util.Objects;

@FunctionalInterface
public interface FloatBinaryOperator {
    float applyAsFloat(float left, float right);

    static FloatBinaryOperator sum() {
        return Float::sum;
    }

    static FloatBinaryOperator subtract() {
        return (float left, float right) -> left - right;
    }

    static FloatBinaryOperator product() {
        return (float left, float right) -> left * right;
    }

    static FloatBinaryOperator divide() {
        return (float left, float right) -> left / right;
    }

    static FloatBinaryOperator min() {
        return Math::min;
    }

    static FloatBinaryOperator max() {
        return Math::max;
    }

    default FloatBinaryOperator accumulate(FloatUnaryOperator rightOperator) {
        Objects.requireNonNull(rightOperator);
        return (float leftAccumulator, float right) -> applyAsFloat(leftAccumulator, rightOperator.applyAsFloat(right));
    }

    default FloatBinaryOperator andThen(FloatUnaryOperator after) {
        Objects.requireNonNull(after);
        return (float left, float right) -> after.applyAsFloat(applyAsFloat(left, right));
    }
}
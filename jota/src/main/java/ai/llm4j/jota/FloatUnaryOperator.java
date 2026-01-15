package ai.llm4j.jota;

import java.util.Objects;

@FunctionalInterface
public interface FloatUnaryOperator {
    static FloatUnaryOperator constant(float value) {
        return unused -> value;
    }

    float applyAsFloat(float operand);

    default FloatUnaryOperator compose(FloatUnaryOperator before) {
        Objects.requireNonNull(before);
        return (float v) -> applyAsFloat(before.applyAsFloat(v));
    }

    default FloatUnaryOperator andThen(FloatUnaryOperator after) {
        Objects.requireNonNull(after);
        return (float t) -> after.applyAsFloat(applyAsFloat(t));
    }

    static FloatUnaryOperator identity() {
        return (float f) -> f;
    }

    static FloatUnaryOperator negate() {
        return (float f) -> -f;
    }

    static FloatUnaryOperator sin() {
        return (float f) -> (float) Math.sin(f);
    }

    static FloatUnaryOperator cos() {
        return (float f) -> (float) Math.cos(f);
    }

    static FloatUnaryOperator sqrt() {
        return (float f) -> (float) Math.sqrt(f);
    }

    static FloatUnaryOperator square() {
        return (float f) -> f * f;
    }

    static FloatUnaryOperator exp() {
        return (float f) -> (float) Math.exp(f);
    }

    static FloatUnaryOperator abs() {
        return Math::abs;
    }

    static FloatUnaryOperator inverse() {
        return (float f) -> 1f / f;
    }
}

package ai.qxotic.span;

@FunctionalInterface
public interface FloatBinaryOperator {

    FloatBinaryOperator SUM = Float::sum;
    FloatBinaryOperator SUM_OF_SQUARES = (acc, value) -> acc + value * value;
    FloatBinaryOperator MUL = (f, s) -> f * s;
    FloatBinaryOperator MIN = Math::min;
    FloatBinaryOperator MAX = Math::max;

    float apply(float first, float second);
}

package com.qxotic.span;

@FunctionalInterface
public interface FloatUnaryOperator {

    float apply(float value);

    FloatUnaryOperator EXP = f -> (float) Math.exp(f);
    FloatUnaryOperator SQRT = f -> (float) Math.sqrt(f);

    FloatUnaryOperator SILU = f -> (float) (f / (1.0 + Math.exp(-f)));

    FloatUnaryOperator GELU =
            f ->
                    (float)
                            (0.5
                                    * f
                                    * (1
                                            + Math.tanh(
                                                    Math.sqrt(2 / Math.PI)
                                                            * (f + 0.044715 * Math.pow(f, 3)))));
    FloatUnaryOperator SIN = f -> (float) Math.sin(f);
    FloatUnaryOperator COS = f -> (float) Math.cos(f);
    FloatUnaryOperator TAN = f -> (float) Math.tan(f);

    FloatUnaryOperator TANH = f -> (float) Math.tanh(f);

    FloatUnaryOperator SOFTPLUS =
            f -> {
                if (f > 20) {
                    return f;
                }
                if (f < -20) {
                    return 0;
                }
                return (float) Math.log1p(Math.exp(f)); // ln(1 + e^x)
            };

    static FloatUnaryOperator XIELU(float alphaP, float alphaN, float beta, float eps) {
        float ap = SOFTPLUS.apply(alphaP);
        float an = beta + SOFTPLUS.apply(alphaN);
        return x ->
                (x > 0)
                        ? (ap * x * x + beta * x)
                        : (float) (an * Math.expm1(Math.min(x, eps))) - an * x + beta * x;
    }
}

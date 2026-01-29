package ai.qxotic.jota.ir.irt;

/**
 * Unary operators for IR-T. Kept separate from the tensor package to maintain IR-T independence.
 */
public enum UnaryOperator {
    NEGATE,
    ABS,
    EXP,
    LOG,
    SQRT,
    SQUARE, // ?
    SIN,
    COS,
    TAN,
    TANH, // for gelu
    RECIPROCAL, // 1  / x
    LOGICAL_NOT,
    BITWISE_NOT
}

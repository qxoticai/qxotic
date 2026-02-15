package ai.qxotic.jota.ir.lir;

public enum LIRExprKind {
    S_CONST,
    S_INPUT,
    S_UNARY,
    S_BINARY,
    S_TERNARY,
    S_CAST,
    S_LOAD,
    S_FROM_INDEX,
    S_REF,
    I_CONST,
    I_VAR,
    I_BINARY,
    I_FROM_SCALAR,
    BLOCK,
    STORE,
    YIELD,
    STRUCTURED_FOR
}

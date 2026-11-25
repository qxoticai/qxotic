package com.qxotic.jota.stablehlo.type;

/** {@code !stablehlo.token}: orders side-effecting operations. */
public record TokenType() implements Type {

    public static final TokenType INSTANCE = new TokenType();

    @Override
    public String mlir() {
        return "!stablehlo.token";
    }

    @Override
    public String toString() {
        return mlir();
    }
}

package com.qxotic.jota.stablehlo.tir;

/** A StableHLO construct that has no TIR translation (yet). */
public final class UnsupportedStableHloException extends RuntimeException {

    private final String operation;

    public UnsupportedStableHloException(String operation, String reason) {
        super(operation + ": " + reason);
        this.operation = operation;
    }

    /** Name of the offending operation ({@code stablehlo.while}...). */
    public String operation() {
        return operation;
    }
}

package com.qxotic.jota.stablehlo.vhlo;

/** A VHLO program that cannot be legalized to StableHLO. */
public final class VhloException extends RuntimeException {

    public VhloException(String message) {
        super(message);
    }
}

package com.qxotic.jota.stablehlo.bytecode;

/** Malformed or unsupported MLIR bytecode. */
public final class BytecodeException extends RuntimeException {

    public BytecodeException(String message) {
        super(message);
    }

    public BytecodeException(String message, Throwable cause) {
        super(message, cause);
    }
}

package com.qxotic.jota.stablehlo.interpret;

/** An operation, type or attribute the interpreter does not support, or a malformed program. */
public class InterpreterException extends RuntimeException {

    public InterpreterException(String message) {
        super(message);
    }
}

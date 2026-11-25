package com.qxotic.jota.stablehlo.verify;

import com.qxotic.jota.stablehlo.ir.Operation;

/**
 * A violated constraint: the operation, the constraint id of {@code docs/spec.md} ({@code "C3"}, or
 * a short tag for structural checks) and a message.
 */
public record Diagnostic(Operation operation, String constraint, String message) {

    @Override
    public String toString() {
        return operation.name() + " (" + constraint + "): " + message;
    }
}

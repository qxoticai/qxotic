package com.qxotic.jota.ir.tir;

import com.qxotic.jota.ir.TIRToLIRLowerer;

/** Validates that a scheduled kernel step can be lowered to LIR. */
public final class LoweringSupportChecker {

    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();

    public void verifyOrThrow(TIRGraph graph, String context) {
        try {
            lowerer.lower(graph);
        } catch (RuntimeException e) {
            throw new UnsupportedOperationException(
                    "Unsupported scheduled kernel " + context + ": " + e.getMessage(), e);
        }
    }
}

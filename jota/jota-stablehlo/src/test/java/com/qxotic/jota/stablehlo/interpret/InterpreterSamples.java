package com.qxotic.jota.stablehlo.interpret;

/** Shared by {@link InterpreterSampleTest} and the opt-in corpus test. */
public final class InterpreterSamples {

    private InterpreterSamples() {}

    /**
     * Runs a sample chunk ({@code name} starting with {@code text/interpret} selects the reference
     * semantics); true when it ran and its checks passed, false when it uses an unsupported
     * feature; a failed check throws {@link Interpreter.CheckFailure}.
     */
    public static boolean runChecks(String name, String text) {
        return InterpreterSampleTest.runChecksImpl(name, text);
    }
}

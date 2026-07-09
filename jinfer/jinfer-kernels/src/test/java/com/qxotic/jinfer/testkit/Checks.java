// The pass/fail protocol shared by every testkit scenario: count failures, print the verdict,
// exit non-zero. One implementation so weights-free scenarios (OracleScenario) and the model
// Harness report identically.
package com.qxotic.jinfer.testkit;

public final class Checks {

    private int failures;

    public void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }

    /**
     * Prints the verdict ({@code name + ": " + allOkMessage}) and exits non-zero on any failure.
     */
    public void finish(String name, String allOkMessage) {
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println(name + ": " + allOkMessage);
    }
}

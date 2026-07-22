// The pass/fail protocol shared by every testkit scenario: count failures, print each verdict,
// and throw at the end. One implementation so weights-free scenarios (OracleScenario) and the
// model Harness report identically under JUnit.
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
     * Prints the verdict ({@code name + ": " + allOkMessage}); throws {@link AssertionError} on any
     * failure so the enclosing JUnit test fails.
     */
    public void finish(String name, String allOkMessage) {
        if (failures > 0) {
            throw new AssertionError(name + ": " + failures + " failure(s) - see FAIL lines above");
        }
        System.out.println(name + ": " + allOkMessage);
    }
}

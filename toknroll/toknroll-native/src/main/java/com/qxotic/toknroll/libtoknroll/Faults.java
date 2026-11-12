package com.qxotic.toknroll.libtoknroll;

/**
 * {@code -Dtoknroll.test.escape=<where>:<kind>}, for the C tests alone: throws at {@code where} an
 * error that no catch sees, as native image may raise an OutOfMemoryError or a StackOverflowError,
 * so the tests reach the entry points' exception handlers on purpose. kind is {@code oom}, {@code
 * stack} or anything else for an exception. Read once; unset, a check is one comparison.
 */
final class Faults {
    private static final String SPEC = System.getProperty("toknroll.test.escape", "");

    static void at(String where) {
        if (SPEC.isEmpty() || !SPEC.startsWith(where + ":")) return;
        switch (SPEC.substring(where.length() + 1)) {
            case "oom" -> throw new OutOfMemoryError("injected at " + where);
            case "stack" -> throw new StackOverflowError("injected at " + where);
            default -> throw new IllegalStateException("injected at " + where);
        }
    }

    private Faults() {}
}

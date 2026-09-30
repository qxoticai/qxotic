package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Runs on every platform: the native calls link, and a captured stderr is no terminal. */
class TerminalTest {

    @Test
    void capturedOutputIsNoTerminal() {
        assertFalse(Terminal.isTerminal(1), "the test runner captures stdout");
        assertFalse(Terminal.isTerminal(2), "the test runner captures stderr");
        assertNull(Terminal.stderr(), "no live view without a terminal");
    }

    @Test
    void columnsAlwaysAnswer() {
        assertTrue(Terminal.columns() >= 1);
    }
}

package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

class TerminalSupportTest {
    @Test
    void capturedStreamsDoNotEnableTerminalOutput() {
        assertFalse(TerminalSupport.isTerminal(1));
        assertFalse(TerminalSupport.isTerminal(2));
        assertFalse(TerminalSupport.enableAnsi(2));
        assertTrue(TerminalSupport.columns(2) > 0);
        assertFalse(TerminalSupport.isTerminal(-1));
        assertFalse(TerminalSupport.isTerminal(3));
        assertFalse(TerminalSupport.enableAnsi(0));
    }

    @Test
    void windowsWidthUsesTheVisibleWindowRatherThanTheBuffer() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = arena.allocate(22, 2);
            info.set(ValueLayout.JAVA_SHORT, 0, (short) 1000); // buffer width
            info.set(ValueLayout.JAVA_SHORT, 10, (short) 17); // window left
            info.set(ValueLayout.JAVA_SHORT, 14, (short) 46); // inclusive right
            assertEquals(30, TerminalSupport.windowsColumns(info));
            info.set(ValueLayout.JAVA_SHORT, 14, (short) 66);
            assertEquals(50, TerminalSupport.windowsColumns(info));
            info.set(ValueLayout.JAVA_SHORT, 14, (short) 96);
            assertEquals(80, TerminalSupport.windowsColumns(info));
            info.set(ValueLayout.JAVA_SHORT, 14, (short) 0);
            assertTrue(TerminalSupport.windowsColumns(info) <= 0);
        }
    }

    @Test
    void unicodeFollowsLocalePrecedence() {
        assertTrue(TerminalSupport.isUtf8(Map.of("LANG", "en_US.UTF-8")));
        assertTrue(TerminalSupport.isUtf8(Map.of("LC_ALL", "C.utf8", "LANG", "C")));
        assertFalse(TerminalSupport.isUtf8(Map.of("LC_ALL", "C", "LANG", "en_US.UTF-8")));
        assertFalse(TerminalSupport.isUtf8(Map.of("LC_CTYPE", "POSIX")));
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void windowsConsoleBindingsAreAvailableEvenWithRedirectedStreams() throws Exception {
        assertTrue(
                TerminalSupport.class.getModule().isNativeAccessEnabled(),
                "the test JVM must enable native access for jinfer-hub");
        for (String name :
                List.of(
                        "GET_STD_HANDLE",
                        "GET_CONSOLE_MODE",
                        "SET_CONSOLE_MODE",
                        "GET_SCREEN_BUFFER_INFO",
                        "GET_OUTPUT_CODE_PAGE",
                        "SET_OUTPUT_CODE_PAGE")) {
            var handle = TerminalSupport.class.getDeclaredField(name);
            handle.setAccessible(true);
            assertNotNull(handle.get(null), name + " could not link kernel32");
        }
    }
}

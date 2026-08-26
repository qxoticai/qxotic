package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.internal.MemoryFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

class MemoryAccessChecksTest {

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {" ", "unknown"})
    void defaultsToRuntimeChecks(String value) {
        assertEquals(MemoryAccessChecks.Mode.RUNTIME, MemoryAccessChecks.resolveMode(value));
    }

    @ParameterizedTest
    @CsvSource({"off, OFF", "ASSERT, ASSERT", "Runtime, RUNTIME"})
    void parsesCheckModeWithoutCaseSensitivity(String value, MemoryAccessChecks.Mode expected) {
        assertEquals(expected, MemoryAccessChecks.resolveMode(value));
    }

    @Test
    void checkBoundsRejectsOverflowingRange() {
        // offset + size wraps negative and used to satisfy "<= byteSize"
        Memory<byte[]> memory = MemoryFactory.ofBytes(new byte[16]);
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> MemoryAccessChecks.checkBounds(memory, 8, Long.MAX_VALUE));
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> MemoryAccessChecks.checkBounds(memory, Long.MAX_VALUE, 8));
        MemoryAccessChecks.checkBounds(memory, 8, 8); // exact fit still passes
    }
}

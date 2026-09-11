package com.qxotic.jam.internal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class MemoryChecksTest {

    @Test
    void nativeRequirementRejectsHeapSegments() {
        try (Arena arena = Arena.ofConfined()) {
            assertDoesNotThrow(() -> MemoryChecks.requireNative(arena.allocate(1), "operand"));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            MemoryChecks.requireNative(
                                    MemorySegment.ofArray(new byte[1]), "operand"));
        }
    }

    @Test
    void nativeRequirementRejectsClosedSegments() {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(1);
        arena.close();

        assertThrows(
                IllegalStateException.class, () -> MemoryChecks.requireNative(segment, "operand"));
    }

    @Test
    void everyDtypeAcceptsOnlyAnInBoundsSpan() {
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3;
            long offset = 7;
            for (GGMLType type : GGMLType.values()) {
                int rowElements = type.elementsPerBlock();
                int stride = 2 * rowElements;
                long required = type.spanBytes(rows, stride, rowElements);
                MemorySegment exact = arena.allocate(offset + required);

                assertDoesNotThrow(
                        () ->
                                MemoryChecks.checkSegment(
                                        type.name(),
                                        exact,
                                        offset,
                                        type.code(),
                                        stride,
                                        rows,
                                        rowElements));
                assertThrows(
                        IndexOutOfBoundsException.class,
                        () ->
                                MemoryChecks.checkSegment(
                                        type.name(),
                                        exact,
                                        offset + 1,
                                        type.code(),
                                        stride,
                                        rows,
                                        rowElements));
                assertThrows(
                        IndexOutOfBoundsException.class,
                        () ->
                                MemoryChecks.checkSegment(
                                        type.name(),
                                        exact,
                                        -1,
                                        type.code(),
                                        stride,
                                        rows,
                                        rowElements));
                assertThrows(
                        IndexOutOfBoundsException.class,
                        () ->
                                MemoryChecks.checkSegment(
                                        type.name(),
                                        exact,
                                        Long.MAX_VALUE,
                                        type.code(),
                                        stride,
                                        rows,
                                        rowElements));
            }
        }
    }

    @Test
    void rejectsByteSpanOverflow() {
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () ->
                            MemoryChecks.checkSegment(
                                    "operand",
                                    arena.allocate(1),
                                    0,
                                    GGMLType.F32.code(),
                                    Integer.MAX_VALUE,
                                    Integer.MAX_VALUE,
                                    1));
        }
    }
}

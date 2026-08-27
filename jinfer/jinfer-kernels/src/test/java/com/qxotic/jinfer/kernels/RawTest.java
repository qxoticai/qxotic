package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Segments;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class RawTest {

    @Test
    void aHeapBackedViewIsRefusedNotDereferenced() {
        // under absolute addressing a heap segment's address() is ~16 inside the array object;
        // every kernel operand passes through Raw, so this is where it must stop
        assumeTrue(Segments.absoluteAddressing());
        var heap = Views.wrap(MemorySegment.ofArray(new float[8]), DataType.FP32, Shape.flat(8));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Raw.f32(heap, "rows"));
        assertTrue(e.getMessage().contains("heap-backed"), e.getMessage());
    }
}

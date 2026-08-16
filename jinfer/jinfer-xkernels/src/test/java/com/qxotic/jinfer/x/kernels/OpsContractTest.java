package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** Small semantic contracts that a differential oracle can accidentally agree on. */
class OpsContractTest {

    @Test
    void argmaxIsRelativeToItsWindowAndKeepsTheFirstTie() {
        try (Arena arena = Arena.ofConfined()) {
            var values =
                    Views.fromFloatArray(
                            new PanamaMemoryArena(arena), new float[] {0, 3, 1, 0, 9, 9, 0, 2});
            assertEquals(1, Ops.argmax(values, 0, 4));
            assertEquals(0, Ops.argmax(values, 4, 4));
            assertEquals(4, Ops.argmax(values, 0, 8));
        }
    }
}

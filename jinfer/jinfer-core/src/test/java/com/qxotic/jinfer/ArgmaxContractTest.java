package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/**
 * Pins argmax's WINDOW-RELATIVE contract: {@code argmax(row * w, w)} is an index in {@code [0, w)}
 * - a token id when the window is a logits row. The absolute contract this replaced (e6eadb03)
 * returned {@code row * vocab + id} from every off-row call: speculative decoding emitted those as
 * garbage tokens and could never accept past its first verify row.
 */
class ArgmaxContractTest {

    @Test
    void argmaxIsRelativeToTheWindow() {
        try (Arena a = Arena.ofShared()) {
            F32FloatTensor t = F32FloatTensor.allocate(a, 8);
            float[] v = {0, 3, 1, 0, 9, 0, 0, 2}; // two 4-wide rows: maxes at rel 1 and rel 0
            for (int i = 0; i < v.length; i++) t.setFloat(i, v[i]);
            assertEquals(1, t.argmax(0, 4));
            assertEquals(0, t.argmax(4, 4)); // the absolute contract returned 4 here
            assertEquals(4, t.argmax()); // whole-tensor overload: unchanged
        }
    }
}

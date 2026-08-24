package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class UtilTest {

    @Test
    void wrapsOneNegativeRange() {
        assertEquals(0, Util.wrapAround(0, 3));
        assertEquals(2, Util.wrapAround(-1, 3));
        assertEquals(0, Util.wrapAround(-3, 3));
    }

    @Test
    void rejectsIndicesOutsideOneWrappedRange() {
        assertThrows(IllegalArgumentException.class, () -> Util.wrapAround(0, 0));
        assertThrows(IllegalArgumentException.class, () -> Util.wrapAround(3, 3));
        assertThrows(IllegalArgumentException.class, () -> Util.wrapAround(-4, 3));
    }
}

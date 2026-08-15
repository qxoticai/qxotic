package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LayoutTileTest {

    @Test
    void tile_matchesLogicalDivideDefinition() {
        Layout base = Layout.rowMajor(24);

        Layout expected = base.logicalDivide(Layout.rowMajor(4));
        Layout actual = base.tile(4);

        assertEquals(expected, actual);
    }

    @Test
    void tile_nestedLayout_matchesLogicalDivideDefinition() {
        Layout base = Layout.rowMajor(Shape.of(2, Shape.of(3, 4)));

        Layout expected = base.logicalDivide(Layout.rowMajor(4));
        Layout actual = base.tile(4);

        assertEquals(expected, actual);
    }

    @Test
    void tile_isDeterministic() {
        Layout base = Layout.rowMajor(16);

        Layout first = base.tile(4);
        Layout second = base.tile(4);

        assertEquals(first, second);
    }

    @Test
    void tile_zeroOrNegativeSize_throwsIllegalArgumentException() {
        Layout base = Layout.rowMajor(8);

        assertThrows(IllegalArgumentException.class, () -> base.tile(0));
        assertThrows(IllegalArgumentException.class, () -> base.tile(-1));
    }

    @Test
    void tile_zeroSizeBase_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(0, 3), Stride.flat(3, 1));
        assertThrows(IllegalArgumentException.class, () -> base.tile(2));
    }

    @Test
    void tile_scalarBase_tileOne_matchesLogicalDivideDefinition() {
        Layout base = Layout.scalar();

        Layout expected = base.logicalDivide(Layout.rowMajor(1));
        Layout actual = base.tile(1);

        assertEquals(expected, actual);
    }
}

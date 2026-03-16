package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LayoutLogicalProductTest {

    @Test
    void logicalProduct_rowMajorOneDimensional_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(4);
        Layout other = Layout.rowMajor(6);

        Layout expected =
                makeLayout(
                        base,
                        base.complement(Math.multiplyExact(base.shape().size(), other.cosize()))
                                .compose(other));

        Layout actual = base.logicalProduct(other);

        assertEquals(expected, actual);
    }

    @Test
    void logicalProduct_nestedBase_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(Shape.of(2, Shape.of(2, 2)));
        Layout other = Layout.rowMajor(3);

        Layout expected =
                makeLayout(
                        base,
                        base.complement(Math.multiplyExact(base.shape().size(), other.cosize()))
                                .compose(other));

        Layout actual = base.logicalProduct(other);

        assertEquals(expected, actual);
    }

    @Test
    void logicalProduct_scalarOther_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(8);
        Layout other = Layout.scalar();

        Layout expected =
                makeLayout(
                        base,
                        base.complement(Math.multiplyExact(base.shape().size(), other.cosize()))
                                .compose(other));

        Layout actual = base.logicalProduct(other);

        assertEquals(expected, actual);
    }

    @Test
    void logicalProduct_isDeterministic() {
        Layout base = Layout.rowMajor(4);
        Layout other = Layout.rowMajor(5);

        Layout first = base.logicalProduct(other);
        Layout second = base.logicalProduct(other);

        assertEquals(first, second);
    }

    @Test
    void logicalProduct_outputMatchesDerivedFormulaByDefinition() {
        Layout base = Layout.rowMajor(6);
        Layout other = Layout.rowMajor(4);

        Layout formula =
                makeLayout(
                        base,
                        base.complement(Math.multiplyExact(base.shape().size(), other.cosize()))
                                .compose(other));

        assertEquals(formula, base.logicalProduct(other));
    }

    @Test
    void logicalProduct_nullOther_throwsNullPointerException() {
        Layout base = Layout.rowMajor(4);
        assertThrows(NullPointerException.class, () -> base.logicalProduct(null));
    }

    @Test
    void logicalProduct_zeroSizeBase_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(0, 3), Stride.flat(3, 1));
        Layout other = Layout.rowMajor(2);

        assertThrows(IllegalArgumentException.class, () -> base.logicalProduct(other));
    }

    @Test
    void logicalProduct_nonInjectiveBase_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));
        Layout other = Layout.rowMajor(2);

        assertThrows(IllegalArgumentException.class, () -> base.logicalProduct(other));
    }

    @Test
    void logicalProduct_negativeStrideBase_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(2, 3), Stride.flat(-3, -1));
        Layout other = Layout.rowMajor(2);

        assertThrows(IllegalArgumentException.class, () -> base.logicalProduct(other));
    }

    @Test
    void logicalProduct_targetOverflow_throwsArithmeticException() {
        Layout base = Layout.rowMajor(Long.MAX_VALUE);
        Layout other = Layout.rowMajor(2);

        assertThrows(ArithmeticException.class, () -> base.logicalProduct(other));
    }

    @Test
    void logicalProduct_otherCosizeOverflow_throwsArithmeticException() {
        Layout base = Layout.rowMajor(4);
        Layout other = Layout.of(Shape.flat(2, 2), Stride.flat(Long.MAX_VALUE, Long.MAX_VALUE));

        assertThrows(ArithmeticException.class, () -> base.logicalProduct(other));
    }

    @Test
    void logicalProduct_resultContainsBaseAsFirstMode() {
        Layout base = Layout.rowMajor(4);
        Layout other = Layout.rowMajor(3);

        Layout result = base.logicalProduct(other);

        assertEquals(base.shape(), result.shape().modeAt(0));
        assertEquals(base.stride(), result.stride().modeAt(0));
    }

    private static Layout makeLayout(Layout left, Layout right) {
        return Layout.of(
                Shape.of(left.shape(), right.shape()), Stride.of(left.stride(), right.stride()));
    }
}

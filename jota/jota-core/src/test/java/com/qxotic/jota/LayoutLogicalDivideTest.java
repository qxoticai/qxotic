package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LayoutLogicalDivideTest {

    @Test
    void logicalDivide_rowMajorOneDimensional_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(24);
        Layout tiler = Layout.rowMajor(4);

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected, actual);
    }

    @Test
    void logicalDivide_nestedLayouts_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(Shape.of(2, Shape.of(3, 4)));
        Layout tiler = Layout.rowMajor(Shape.of(2, 2));

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected, actual);
    }

    @Test
    void logicalDivide_nonStandardRepresentableBase_matchesDerivedFormula() {
        Layout base = Layout.of(Shape.flat(2, 2, 2), Stride.flat(8, 2, 1));
        Layout tiler = Layout.rowMajor(4);

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected, actual);
    }

    @Test
    void logicalDivide_tilerWithSingletons_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(24);
        Layout tiler = Layout.of(Shape.flat(1, 4, 1), Stride.flat(0, 1, 0));

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected, actual);
    }

    @Test
    void logicalDivide_scalarTiler_matchesDerivedFormula() {
        Layout base = Layout.rowMajor(8);
        Layout tiler = Layout.scalar();

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected, actual);
    }

    @Test
    void logicalDivide_isDeterministic() {
        Layout base = Layout.rowMajor(16);
        Layout tiler = Layout.rowMajor(4);

        Layout first = base.logicalDivide(tiler);
        Layout second = base.logicalDivide(tiler);

        assertEquals(first, second);
    }

    @Test
    void logicalDivide_nullTiler_throwsNullPointerException() {
        Layout base = Layout.rowMajor(8);
        assertThrows(NullPointerException.class, () -> base.logicalDivide(null));
    }

    @Test
    void logicalDivide_zeroSizeBase_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(0, 3), Stride.flat(3, 1));
        Layout tiler = Layout.rowMajor(2);

        assertThrows(IllegalArgumentException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_composeDomainMismatch_throwsIllegalArgumentException() {
        Layout base = Layout.rowMajor(5);
        Layout tiler = Layout.rowMajor(6);

        assertThrows(IllegalArgumentException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_nonInjectiveTiler_throwsIllegalArgumentException() {
        Layout base = Layout.rowMajor(12);
        Layout tiler = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));

        assertThrows(IllegalArgumentException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_negativeStrideTiler_throwsIllegalArgumentException() {
        Layout base = Layout.rowMajor(12);
        Layout tiler = Layout.of(Shape.flat(2, 3), Stride.flat(-3, -1));

        assertThrows(IllegalArgumentException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_nonRepresentableCompose_throwsIllegalArgumentException() {
        Layout base = Layout.of(Shape.flat(2, 3), Stride.flat(1, 2));
        Layout tiler = Layout.rowMajor(6);

        assertThrows(IllegalArgumentException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_complementOverflow_throwsArithmeticException() {
        Layout base = Layout.rowMajor(24);
        Layout tiler = Layout.of(Shape.flat(2), Stride.flat(Long.MAX_VALUE));

        assertThrows(ArithmeticException.class, () -> base.logicalDivide(tiler));
    }

    @Test
    void logicalDivide_outputShapeMatchesDerivedFormula() {
        Layout base = Layout.rowMajor(24);
        Layout tiler = Layout.rowMajor(4);

        Layout expected =
                base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));
        Layout actual = base.logicalDivide(tiler);

        assertEquals(expected.shape(), actual.shape());
        assertEquals(expected.stride(), actual.stride());
    }

    @Test
    void logicalDivide_isDerivedFormulaByDefinition() {
        Layout base = Layout.rowMajor(12);
        Layout tiler = Layout.rowMajor(3);

        Layout actual = base.logicalDivide(tiler);
        Layout formula = base.compose(makeLayout(tiler, tiler.complement(base.shape().size())));

        assertTrue(actual.equals(formula));
    }

    private static Layout makeLayout(Layout left, Layout right) {
        return Layout.of(
                Shape.of(left.shape(), right.shape()), Stride.of(left.stride(), right.stride()));
    }
}

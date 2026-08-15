package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The logical/physical shape laws of the {@link DataType} contract: blocking tiles only the last
 * dim in flatten order, the conversions round-trip both directions, nesting is preserved,
 * divisibility is enforced with an IAE, dense dtypes are identity - and {@link
 * View#logicalShape()}/{@link View#logicalSize()} read the same law at view level.
 */
class DataTypeBlockShapeTest {

    private static final DataType[] BLOCK_DTYPES = {
        DataType.Q4_0,
        DataType.Q4_1,
        DataType.Q5_1,
        DataType.Q8_0,
        DataType.Q4_K,
        DataType.Q5_K,
        DataType.Q6_K,
        DataType.MXFP4,
        DataType.NVFP4,
        DataType.Q1_0,
        DataType.TQ1_0,
        DataType.TQ2_0
    };

    @Test
    void flatRoundTripForEveryBlockDtype() {
        for (DataType dt : BLOCK_DTYPES) {
            long epb = dt.elementsPerBlock();
            Shape logical = Shape.flat(7, 3 * epb);
            Shape physical = dt.physicalShape(logical);
            assertEquals(Shape.flat(7, 3), physical, dt.name());
            assertEquals(logical, dt.logicalShape(physical), dt.name() + " logical round-trip");
            assertEquals(
                    physical,
                    dt.physicalShape(dt.logicalShape(physical)),
                    dt.name() + " physical round-trip");
        }
    }

    @Test
    void nestedShapesScaleOnlyTheLastDim() {
        // Q4_K: 256 logical elements per block
        Shape logical = Shape.of(2, Shape.of(3, 512));
        Shape physical = DataType.Q4_K.physicalShape(logical);
        // the nesting is preserved and ONLY the last dim was scaled - not every leaf of the
        // last mode (that naive reading would give (2, (768, 2)) and 4x the storage)
        assertEquals(Shape.of(2, Shape.of(3, 2)), physical);
        assertEquals(logical, DataType.Q4_K.logicalShape(physical), "nested round-trip");
        // a non-innermost dim is never touched, even when not divisible by the block size
        assertEquals(Shape.flat(100, 1), DataType.Q8_0.physicalShape(Shape.flat(100, 32)));
    }

    @Test
    void divisibilityIsEnforcedOnTheLastDim() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DataType.Q8_0.physicalShape(Shape.flat(4, 100)));
        assertTrue(e.getMessage().contains("100"), e.getMessage());
        assertTrue(e.getMessage().contains("32"), e.getMessage());
        assertTrue(e.getMessage().contains("q8_0"), e.getMessage());
        // nested: the check fires on the last dim in flatten order
        assertThrows(
                IllegalArgumentException.class,
                () -> DataType.Q8_0.physicalShape(Shape.of(2, Shape.of(3, 100))));
    }

    @Test
    void denseDtypesAreIdentityEvenForNestedShapes() {
        Shape nested = Shape.of(2, Shape.of(3, 5));
        assertEquals(nested, DataType.FP32.physicalShape(nested));
        assertEquals(nested, DataType.FP32.logicalShape(nested));
        assertEquals(nested, DataType.I8.physicalShape(nested));
    }

    @Test
    void denseScalarsAreIdentity() {
        assertEquals(Shape.scalar(), DataType.FP32.physicalShape(Shape.scalar()));
        assertEquals(Shape.scalar(), DataType.FP32.logicalShape(Shape.scalar()));
        assertEquals(Shape.scalar(), DataType.BOOL.physicalShape(Shape.scalar()));
    }

    @Test
    void blockDtypeScalarsAreNotRepresentable() {
        // rank 0 has no innermost axis: one block holds epb elements, never 1 - reject, loudly
        for (DataType dt : BLOCK_DTYPES) {
            IllegalArgumentException p =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> dt.physicalShape(Shape.scalar()),
                            dt.name() + " physicalShape");
            assertTrue(p.getMessage().contains(dt.name()), p.getMessage());
            assertTrue(
                    p.getMessage().contains(String.valueOf(dt.elementsPerBlock())), p.getMessage());
            assertThrows(
                    IllegalArgumentException.class,
                    () -> dt.logicalShape(Shape.scalar()),
                    dt.name() + " logicalShape");
        }
    }

    @Test
    void byteSizeForCountsStorageUnits() {
        // Q8_0: one storage unit is a 34-byte block holding 32 logical elements
        assertEquals(34, DataType.Q8_0.byteSizeFor(1));
        assertEquals(68, DataType.Q8_0.byteSizeFor(2));
        // the bytes of a logically-dimensioned tensor: fold first, then count
        Shape physical = DataType.Q8_0.physicalShape(Shape.flat(7, 64));
        assertEquals(7L * 2 * 34, DataType.Q8_0.byteSizeFor(physical));
        assertEquals(4, DataType.FP32.byteSizeFor(1)); // dense: unit == element
    }

    @Test
    void viewLevelLogicalShapeAndSize() {
        // View is an interface; a stub pins the default methods without a backing MemoryView
        View view = stubView(Shape.flat(7, 2), DataType.Q8_0); // 7x2 blocks
        assertEquals(Shape.flat(7, 64), view.logicalShape());
        assertEquals(7 * 64, view.logicalSize());
        View dense = stubView(Shape.flat(7, 64), DataType.FP32);
        assertEquals(Shape.flat(7, 64), dense.logicalShape());
    }

    private static View stubView(Shape shape, DataType dt) {
        return new View() {
            @Override
            public Layout layout() {
                return Layout.rowMajor(shape);
            }

            @Override
            public DataType dataType() {
                return dt;
            }

            @Override
            public long byteOffset() {
                return 0;
            }
        };
    }
}

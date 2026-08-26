package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.internal.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Block-quantized dtypes count storage BLOCKS in {@code shape()}; {@code logicalShape()} maps back
 * to element dims by scaling the innermost axis by {@code elementsPerBlock()}. For every other
 * dtype physical == logical.
 */
public class LogicalShapeTest {

    @Test
    void identityForScalarDtypes() {
        Shape s = Shape.flat(2048, 2048);
        assertSame(s, DataType.FP32.logicalShape(s));
        assertSame(s, DataType.FP16.logicalShape(s));
        assertSame(s, DataType.I8.logicalShape(s));
        assertSame(s, DataType.FP32.physicalShape(s));
    }

    @Test
    void blockDtypeScalesInnermostAxis() {
        assertEquals(Shape.flat(2048), DataType.Q8_0.logicalShape(Shape.flat(64)));
        assertEquals(Shape.flat(2048, 2048), DataType.Q8_0.logicalShape(Shape.flat(2048, 64)));
        assertEquals(Shape.flat(2048, 2048), DataType.Q4_0.logicalShape(Shape.flat(2048, 64)));
        // a block-quantized scalar is not representable: rank 0 has no innermost axis to tile
        assertThrows(
                IllegalArgumentException.class, () -> DataType.Q8_0.logicalShape(Shape.scalar()));
    }

    @Test
    void physicalShapeIsTheExactInverse() {
        Shape logical = Shape.flat(2048, 2048);
        assertEquals(Shape.flat(2048, 64), DataType.Q8_0.physicalShape(logical));
        assertEquals(logical, DataType.Q8_0.logicalShape(DataType.Q8_0.physicalShape(logical)));
    }

    @Test
    void physicalShapeRequiresDivisibleInnermost() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DataType.Q8_0.physicalShape(Shape.flat(100)));
        assertEquals(true, e.getMessage().contains("100"));
    }

    /** Every block dtype declared on {@link DataType} - a new one joins automatically. */
    static Stream<DataType> blockDtypes() {
        return Arrays.stream(DataType.class.getFields())
                .filter(field -> field.getType() == DataType.class)
                .map(
                        field -> {
                            try {
                                return (DataType) field.get(null);
                            } catch (IllegalAccessException e) {
                                throw new IllegalStateException(e);
                            }
                        })
                .filter(dtype -> dtype.elementsPerBlock() > 1);
    }

    @ParameterizedTest
    @MethodSource("blockDtypes")
    void logicalShapeIsTheExactInverseForEveryBlockDtype(DataType dtype) {
        long epb = dtype.elementsPerBlock();

        Shape flat = Shape.flat(2048, epb * 7);
        assertEquals(Shape.flat(2048, 7), dtype.physicalShape(flat));
        assertEquals(flat, dtype.logicalShape(dtype.physicalShape(flat)));

        Shape nested = Shape.of(2, Shape.of(3, epb * 5));
        assertEquals(Shape.of(2, Shape.of(3, 5)), dtype.physicalShape(nested));
        assertEquals(nested, dtype.logicalShape(dtype.physicalShape(nested)));
    }

    @Test
    void scalarBehavior() {
        // dense dtypes: identity, reference-identical, scalars included
        assertSame(Shape.scalar(), DataType.FP32.logicalShape(Shape.scalar()));
        assertSame(Shape.scalar(), DataType.FP32.physicalShape(Shape.scalar()));
        // block dtypes: a scalar is not representable on either conversion direction
        assertThrows(
                IllegalArgumentException.class, () -> DataType.Q8_0.physicalShape(Shape.scalar()));
    }

    @Test
    void nestedShapePreservedForBlockDtypes() {
        Shape nested = Shape.of(2, Shape.of(3, 4));
        // nested structure is preserved; only the last dim in flatten order is scaled
        assertEquals(Shape.of(2, Shape.of(3, 128)), DataType.Q8_0.logicalShape(nested));
        assertEquals(nested, DataType.Q8_0.physicalShape(Shape.of(2, Shape.of(3, 128))));
        // identity dtypes never inspect the shape
        assertSame(nested, DataType.FP32.logicalShape(nested));
    }

    @Test
    void memoryViewExposesLogicalShape() {
        MemorySegment seg = Arena.ofAuto().allocate(2048L * 64 * 34);
        Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(seg);
        MemoryView<MemorySegment> weights =
                MemoryView.rowMajor(mem, DataType.Q8_0, Shape.flat(2048, 64));
        assertEquals(Shape.flat(2048, 64), weights.shape());
        assertEquals(Shape.flat(2048, 2048), weights.logicalShape());
        assertEquals(2048L * 2048, weights.logicalSize());
    }
}

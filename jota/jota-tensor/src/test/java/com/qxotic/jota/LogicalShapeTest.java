package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

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

package com.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class TensorEntryTest {

    @Test
    public void testWithOffsetCreatesNewInstance() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertNotSame(original, modified);
    }

    @Test
    public void testWithOffsetPreservesName() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertEquals("weights", modified.name());
    }

    @Test
    public void testWithOffsetPreservesDtype() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertEquals(DType.F32, modified.dtype());
    }

    @Test
    public void testWithOffsetPreservesShape() {
        long[] shape = new long[] {10, 20, 30};
        TensorEntry original = TensorEntry.create("weights", DType.F32, shape, 100);
        TensorEntry modified = original.withOffset(200);

        assertArrayEquals(shape, modified.shape());
    }

    @Test
    public void testWithOffsetChangesOffset() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertEquals(200, modified.byteOffset());
        assertEquals(100, original.byteOffset()); // Original unchanged
    }

    @Test
    public void testWithOffsetZero() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10}, 100);
        TensorEntry modified = original.withOffset(0);

        assertEquals(0, modified.byteOffset());
        assertEquals("weights", modified.name());
        assertEquals(DType.F32, modified.dtype());
    }

    @Test
    public void testWithOffsetNegative() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10}, 100);
        TensorEntry modified = original.withOffset(-50);

        assertEquals(-50, modified.byteOffset());
    }

    @Test
    public void testWithOffsetDifferentDtypes() {
        for (DType dtype : DType.values()) {
            TensorEntry original = TensorEntry.create("tensor", dtype, new long[] {5}, 100);
            TensorEntry modified = original.withOffset(200);

            assertEquals(dtype, modified.dtype(), "Failed for dtype: " + dtype);
        }
    }

    @Test
    public void testWithOffsetPreservesByteSize() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertEquals(original.byteSize(), modified.byteSize());
    }

    @Test
    public void testWithOffsetPreservesTotalElements() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        TensorEntry modified = original.withOffset(200);

        assertEquals(original.totalNumberOfElements(), modified.totalNumberOfElements());
    }

    @Test
    public void testWithOffsetOriginalUnchanged() {
        String originalName = "original_weights";
        DType originalDtype = DType.F64;
        long[] originalShape = new long[] {5, 10, 15};
        long originalOffset = 1024;

        TensorEntry original =
                TensorEntry.create(originalName, originalDtype, originalShape, originalOffset);
        TensorEntry modified = original.withOffset(2048);

        // Verify original is unchanged
        assertEquals(originalName, original.name());
        assertEquals(originalDtype, original.dtype());
        assertArrayEquals(originalShape, original.shape());
        assertEquals(originalOffset, original.byteOffset());

        // Verify modified has new offset
        assertEquals(2048, modified.byteOffset());
    }

    @Test
    public void testWithOffsetEquality() {
        TensorEntry entry1 = TensorEntry.create("weights", DType.F32, new long[] {10}, 100);
        TensorEntry entry2 = TensorEntry.create("weights", DType.F32, new long[] {10}, 200);

        TensorEntry modified = entry1.withOffset(200);

        assertEquals(entry2, modified);
        assertEquals(entry2.hashCode(), modified.hashCode());
    }

    @Test
    public void testWithOffsetChaining() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10}, 100);
        TensorEntry modified = original.withOffset(200).withOffset(300);

        assertEquals(300, modified.byteOffset());
    }

    @Test
    public void testWithOffsetScalar() {
        TensorEntry original = TensorEntry.create("scalar", DType.F32, new long[] {}, 0);
        TensorEntry modified = original.withOffset(64);

        assertEquals(64, modified.byteOffset());
        assertArrayEquals(new long[] {}, modified.shape());
    }

    @Test
    public void testWithOffsetLargeValue() {
        TensorEntry original = TensorEntry.create("weights", DType.F32, new long[] {10}, 0);
        TensorEntry modified = original.withOffset(Long.MAX_VALUE);

        assertEquals(Long.MAX_VALUE, modified.byteOffset());
    }

    @Test
    public void testToString() {
        TensorEntry entry = TensorEntry.create("weights", DType.F32, new long[] {10, 20}, 100);
        String str = entry.toString();
        assertTrue(str.contains("name=weights"));
        assertTrue(str.contains("dtype=F32"));
        assertTrue(str.contains("shape=[10, 20]"));
        assertTrue(str.contains("offset=0x64")); // 100 in hex
        assertTrue(str.contains("byteSize=800"));
    }

    @Test
    public void testToStringScalar() {
        TensorEntry entry = TensorEntry.create("bias", DType.F64, new long[] {}, 0);
        String str = entry.toString();
        assertTrue(str.contains("name=bias"));
        assertTrue(str.contains("dtype=F64"));
        assertTrue(str.contains("shape=[]"));
        assertTrue(str.contains("offset=0x0"));
    }

    @Test
    public void testEqualsSelf() {
        TensorEntry entry = TensorEntry.create("a", DType.F32, new long[] {1}, 0);
        assertEquals(entry, entry);
    }

    @Test
    public void testEqualsNull() {
        TensorEntry entry = TensorEntry.create("a", DType.F32, new long[] {1}, 0);
        assertNotEquals(entry, null);
    }

    @Test
    public void testEqualsWrongType() {
        TensorEntry entry = TensorEntry.create("a", DType.F32, new long[] {1}, 0);
        assertNotEquals(entry, "not a tensor");
    }

    @Test
    public void testEqualsDifferentName() {
        TensorEntry a = TensorEntry.create("x", DType.F32, new long[] {1}, 0);
        TensorEntry b = TensorEntry.create("y", DType.F32, new long[] {1}, 0);
        assertNotEquals(a, b);
    }

    @Test
    public void testEqualsDifferentDtype() {
        TensorEntry a = TensorEntry.create("x", DType.F32, new long[] {1}, 0);
        TensorEntry b = TensorEntry.create("x", DType.F64, new long[] {1}, 0);
        assertNotEquals(a, b);
    }

    @Test
    public void testEqualsDifferentShape() {
        TensorEntry a = TensorEntry.create("x", DType.F32, new long[] {2, 3}, 0);
        TensorEntry b = TensorEntry.create("x", DType.F32, new long[] {3, 2}, 0);
        assertNotEquals(a, b);
    }

    @Test
    public void testEqualsDifferentOffset() {
        TensorEntry a = TensorEntry.create("x", DType.F32, new long[] {1}, 0);
        TensorEntry b = TensorEntry.create("x", DType.F32, new long[] {1}, 100);
        assertNotEquals(a, b);
    }
}

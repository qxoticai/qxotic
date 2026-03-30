package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class TensorEntryTest {

    @Test
    public void testWithOffset() {
        TensorEntry original =
                TensorEntry.create("weights", new long[] {1024, 1024}, GGMLType.F32, 0);
        TensorEntry moved = original.withOffset(4096);

        assertEquals(0, original.offset());
        assertEquals(4096, moved.offset());

        assertEquals(original.name(), moved.name());
        assertArrayEquals(original.shape(), moved.shape());
        assertEquals(original.ggmlType(), moved.ggmlType());
        assertEquals(original.type(), moved.type());
    }

    @Test
    public void testToStringFormat() {
        TensorEntry entry =
                TensorEntry.create("token_embd.weight", new long[] {768, 32000}, GGMLType.F32, 0);
        String str = entry.toString();

        assertTrue(str.contains("TensorEntry{"));
        assertTrue(str.contains("name=token_embd.weight"));
        assertTrue(str.contains("shape=[768, 32000]"));
        assertTrue(str.contains("ggmlType=F32"));
        assertTrue(str.contains("offset=0x0"));
    }

    @Test
    public void testToStringWithHexOffset() {
        TensorEntry entry = TensorEntry.create("test", new long[] {10}, GGMLType.F32, 4660);
        String str = entry.toString();

        assertTrue(str.contains("offset=0x1234"));
    }

    @Test
    public void testByteSize() {
        TensorEntry f32Entry = TensorEntry.create("w", new long[] {10, 20}, GGMLType.F32, 0);
        assertEquals(800, f32Entry.byteSize()); // 10 * 20 * 4 bytes

        TensorEntry q4Entry = TensorEntry.create("w", new long[] {32}, GGMLType.Q4_0, 0);
        assertEquals(18, q4Entry.byteSize()); // One block of 32 elements

        TensorEntry q4Large = TensorEntry.create("w", new long[] {64, 32}, GGMLType.Q4_0, 0);
        assertEquals(1152, q4Large.byteSize()); // 64 blocks * 18 bytes
    }

    @Test
    public void testTypeAlias() {
        TensorEntry entry = TensorEntry.create("w", new long[] {10}, GGMLType.F16, 0);
        assertEquals(entry.ggmlType(), entry.type());
        assertSame(entry.ggmlType(), entry.type());
    }

    @Test
    public void testEqualsAndHashCode() {
        TensorEntry entry1 =
                TensorEntry.create("weights", new long[] {100, 200}, GGMLType.F32, 1024);
        TensorEntry entry2 =
                TensorEntry.create("weights", new long[] {100, 200}, GGMLType.F32, 1024);
        TensorEntry entry3 =
                TensorEntry.create("weights", new long[] {100, 200}, GGMLType.F32, 2048);
        TensorEntry entry4 =
                TensorEntry.create("biases", new long[] {100, 200}, GGMLType.F32, 1024);

        assertEquals(entry1, entry2);
        assertEquals(entry1.hashCode(), entry2.hashCode());

        assertNotEquals(entry1, entry3);
        assertNotEquals(entry1, entry4);
        assertNotEquals(entry1, null);
        assertNotEquals(entry1, "not a tensor");
    }

    @Test
    public void testShapeDefensiveCopy() {
        long[] shape = new long[] {100, 200};
        TensorEntry entry = TensorEntry.create("w", shape, GGMLType.F32, 0);

        shape[0] = 999;
        assertArrayEquals(new long[] {100, 200}, entry.shape());

        long[] retrieved = entry.shape();
        retrieved[0] = 888;
        assertArrayEquals(new long[] {100, 200}, entry.shape());
    }

    @Test
    public void testEmptyShape() {
        TensorEntry entry = TensorEntry.create("scalar", new long[0], GGMLType.F32, 0);

        assertEquals(0, entry.shape().length);
        // Empty shape has element count of 1 (product of empty array is 1)
        assertEquals(4, entry.byteSize()); // 1 element * 4 bytes
    }

    @Test
    public void testScalarShape() {
        // Single-element tensor (scalar-like)
        TensorEntry entry = TensorEntry.create("scalar", new long[] {1}, GGMLType.F32, 0);

        assertArrayEquals(new long[] {1}, entry.shape());
        assertEquals(4, entry.byteSize()); // 1 element * 4 bytes
    }

    @Test
    public void testZeroOffset() {
        TensorEntry entry = TensorEntry.create("test", new long[] {10}, GGMLType.F32, 0);

        assertEquals(0, entry.offset());
        String str = entry.toString();
        assertTrue(str.contains("offset=0x0"));
    }

    @Test
    public void testLargeOffset() {
        long largeOffset = 0x123456789ABCDEF0L;
        TensorEntry entry = TensorEntry.create("test", new long[] {10}, GGMLType.F32, largeOffset);

        assertEquals(largeOffset, entry.offset());
        String str = entry.toString();
        assertTrue(str.contains("offset=0x"));
    }
}

package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** Tests for GGUF interface default methods and edge cases. */
public class GGUFDefaultMethodsTest {

    @Test
    public void testAbsoluteOffsetNullThrowsNpe() {
        GGUF gguf = Builder.newBuilder().build();

        assertThrows(NullPointerException.class, () -> gguf.absoluteOffset(null));
    }

    @Test
    public void testAbsoluteOffsetCalculation() {
        TensorEntry tensor = TensorEntry.create("w", new long[] {10}, GGMLType.F32, 256);
        GGUF gguf = Builder.newBuilder().putTensor(tensor).build();

        long absolute = gguf.absoluteOffset(tensor);

        // Should be tensorDataOffset + tensor.offset()
        assertEquals(gguf.getTensorDataOffset() + 256, absolute);
    }

    @Test
    public void testContainsKeyNullReturnsFalse() {
        GGUF gguf = Builder.newBuilder().putString("key", "value").build();

        // Null key should return false (not throw)
        assertFalse(gguf.containsKey(null));
    }

    @Test
    public void testContainsTensorNullReturnsFalse() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("t", new long[] {1}, GGMLType.F32, 0))
                        .build();

        // Null tensor name should return false (not throw)
        assertFalse(gguf.containsTensor(null));
    }

    @Test
    public void testGetAlignmentWithDefault() {
        GGUF gguf = Builder.newBuilder().build();

        // Default alignment is 32
        assertEquals(32, gguf.getAlignment());
    }

    @Test
    public void testGetAlignmentWithExplicitValue() {
        GGUF gguf = Builder.newBuilder().setAlignment(64).build();

        assertEquals(64, gguf.getAlignment());
    }

    @Test
    public void testDefaultToString() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("name", "test")
                        .putTensor(TensorEntry.create("w", new long[] {10}, GGMLType.F32, 0))
                        .build();

        // Default toString() should return summary format
        String str = gguf.toString();

        assertNotNull(str);
        assertTrue(str.contains("GGUF {"));
        assertTrue(str.contains("version:"));
    }

    @Test
    public void testGetValueOrDefaultWithExistingKey() {
        GGUF gguf = Builder.newBuilder().putString("exists", "value").build();

        assertEquals("value", gguf.getValueOrDefault(String.class, "exists", "default"));
    }

    @Test
    public void testGetValueOrDefaultWithMissingKey() {
        GGUF gguf = Builder.newBuilder().build();

        assertEquals("default", gguf.getValueOrDefault(String.class, "missing", "default"));
        assertNull(gguf.getValueOrDefault(String.class, "missing", null));
    }

    @Test
    public void testGetStringConvenienceMethod() {
        GGUF gguf = Builder.newBuilder().putString("key", "value").build();

        assertEquals("value", gguf.getString("key"));
    }

    @Test
    public void testGetStringOrDefault() {
        GGUF gguf = Builder.newBuilder().putString("key", "value").build();

        assertEquals("value", gguf.getStringOrDefault("key", "default"));
        assertEquals("default", gguf.getStringOrDefault("missing", "default"));
    }
}

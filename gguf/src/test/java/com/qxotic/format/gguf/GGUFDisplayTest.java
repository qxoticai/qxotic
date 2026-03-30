package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class GGUFDisplayTest {

    @Test
    public void testToStringFull() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "test-model")
                        .putInteger("test.value", 42)
                        .putTensor(TensorEntry.create("weights", new long[] {100}, GGMLType.F32, 0))
                        .build();

        String full = gguf.toString(true, true);

        assertNotNull(full);
        assertFalse(full.isEmpty());
        assertTrue(full.contains("version:"));
        assertTrue(full.contains("alignment:"));
        assertTrue(full.contains("general.name"));
        assertTrue(full.contains("test.value"));
        assertTrue(full.contains("weights"));
        // Check it contains the formatted structure
        assertTrue(full.contains("GGUF {"));
        assertTrue(full.contains("metadata:"));
        assertTrue(full.contains("tensors:"));
    }

    @Test
    public void testToStringSummaryOnly() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "test-model")
                        .putTensor(TensorEntry.create("weights", new long[] {100}, GGMLType.F32, 0))
                        .build();

        String summary = gguf.toString(false, false);

        assertNotNull(summary);
        assertFalse(summary.isEmpty());
        assertTrue(summary.contains("GGUF {"));
        // Should show counts but not details
        assertTrue(summary.contains("2") || summary.contains("metadata:"));
        assertTrue(summary.contains("1") || summary.contains("tensors:"));
    }

    @Test
    public void testToStringMetadataOnly() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "test-model")
                        .putTensor(TensorEntry.create("weights", new long[] {100}, GGMLType.F32, 0))
                        .build();

        String metaOnly = gguf.toString(true, false);

        assertNotNull(metaOnly);
        assertFalse(metaOnly.isEmpty());
        assertTrue(metaOnly.contains("general.name"));
        // Should not contain tensor details
        assertFalse(metaOnly.contains("F32[100]"));
    }

    @Test
    public void testToStringTensorsOnly() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "test-model")
                        .putTensor(TensorEntry.create("weights", new long[] {100}, GGMLType.F32, 0))
                        .build();

        String tensorsOnly = gguf.toString(false, true);

        assertNotNull(tensorsOnly);
        assertFalse(tensorsOnly.isEmpty());
        assertTrue(tensorsOnly.contains("weights"));
        assertTrue(tensorsOnly.contains("F32"));
        // Should not contain metadata details
        assertFalse(tensorsOnly.contains("test-model"));
    }

    @Test
    public void testToStringWithElision() {
        int[] largeArray = new int[100];
        for (int i = 0; i < 100; i++) {
            largeArray[i] = i;
        }

        GGUF gguf =
                Builder.newBuilder()
                        .putArrayOfInteger("large", largeArray)
                        .putString("long_text", "a".repeat(200))
                        .build();

        String elided = gguf.toString(true, true, 4, 50);

        assertNotNull(elided);
        // Should contain elipsis for large arrays or long strings
        assertTrue(elided.contains("..."));
    }

    @Test
    public void testToStringEmptyGGUF() {
        GGUF empty = Builder.newBuilder().build();

        String str = empty.toString(true, true);

        assertNotNull(str);
        assertFalse(str.isEmpty());
        assertTrue(str.contains("GGUF {"));
        assertTrue(str.contains("version:"));
        assertTrue(str.contains("alignment:"));
    }

    @Test
    public void testToStringSpecialCharacters() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("with_newline", "line1\nline2")
                        .putString("with_tab", "col1\tcol2")
                        .putString("with_quote", "say \"hello\"")
                        .build();

        String str = gguf.toString(true, false);

        assertNotNull(str);
        // Check that special characters are escaped or the string contains the key
        assertTrue(str.contains("with_newline"));
        assertTrue(str.contains("with_tab"));
        assertTrue(str.contains("with_quote"));
    }

    @Test
    public void testToStringMultipleTensors() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("w1", new long[] {64}, GGMLType.F32, 0))
                        .putTensor(TensorEntry.create("w2", new long[] {64}, GGMLType.F16, 256))
                        .putTensor(TensorEntry.create("w3", new long[] {64}, GGMLType.Q4_0, 384))
                        .build();

        String str = gguf.toString(false, true);

        assertNotNull(str);
        assertFalse(str.isEmpty());
        assertTrue(str.contains("w1"));
        assertTrue(str.contains("w2"));
        assertTrue(str.contains("w3"));
        assertTrue(str.contains("F32"));
        assertTrue(str.contains("F16"));
        assertTrue(str.contains("Q4_0"));
    }

    @Test
    public void testEmptyArrayDisplay() {
        GGUF gguf = Builder.newBuilder().putArrayOfInteger("empty_array", new int[0]).build();

        String str = gguf.toString(true, false);

        assertNotNull(str);
        assertTrue(str.contains("[]"));
    }

    @Test
    public void testArrayAtExactBoundary() {
        // Array with exactly 5 elements (default maxArrayElements)
        int[] fiveElements = new int[] {1, 2, 3, 4, 5};

        GGUF gguf = Builder.newBuilder().putArrayOfInteger("five", fiveElements).build();

        String str = gguf.toString(true, false);

        assertNotNull(str);
        // Should show all elements without elision
        assertFalse(str.contains("..."));
        assertTrue(str.contains("1"));
        assertTrue(str.contains("5"));
    }

    @Test
    public void testStringTruncationBoundary() {
        // String with exactly 50 characters (default maxStringLength)
        String exactly50 = "x".repeat(50);

        GGUF gguf = Builder.newBuilder().putString("boundary", exactly50).build();

        String str = gguf.toString(true, false);

        assertNotNull(str);
        // Should show full string without truncation
        assertFalse(str.contains("..."));
        assertTrue(str.contains(exactly50));
    }

    @Test
    public void testStringTruncationOverBoundary() {
        // String with 51 characters (over default maxStringLength)
        String over50 = "x".repeat(51);

        GGUF gguf = Builder.newBuilder().putString("over_boundary", over50).build();

        String str = gguf.toString(true, false);

        assertNotNull(str);
        // Should be truncated with elipsis
        assertTrue(str.contains("..."));
    }

    @Test
    public void testScalarTensor() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("scalar", new long[] {1}, GGMLType.F32, 0))
                        .build();

        String str = gguf.toString(false, true);

        assertNotNull(str);
        assertTrue(str.contains("scalar"));
        assertTrue(str.contains("F32[1]"));
    }

    @Test
    public void testEmptyShapeTensor() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("empty_shape", new long[0], GGMLType.F32, 0))
                        .build();

        String str = gguf.toString(false, true);

        assertNotNull(str);
        assertTrue(str.contains("empty_shape"));
        assertTrue(str.contains("F32[]"));
    }
}

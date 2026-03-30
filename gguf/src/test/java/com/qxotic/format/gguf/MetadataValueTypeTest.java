package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class MetadataValueTypeTest {

    @Test
    public void testFromIndexValid() {
        assertEquals(MetadataValueType.UINT8, MetadataValueType.fromIndex(0));
        assertEquals(MetadataValueType.INT8, MetadataValueType.fromIndex(1));
        assertEquals(MetadataValueType.UINT16, MetadataValueType.fromIndex(2));
        assertEquals(MetadataValueType.INT16, MetadataValueType.fromIndex(3));
        assertEquals(MetadataValueType.UINT32, MetadataValueType.fromIndex(4));
        assertEquals(MetadataValueType.INT32, MetadataValueType.fromIndex(5));
        assertEquals(MetadataValueType.FLOAT32, MetadataValueType.fromIndex(6));
        assertEquals(MetadataValueType.BOOL, MetadataValueType.fromIndex(7));
        assertEquals(MetadataValueType.STRING, MetadataValueType.fromIndex(8));
        assertEquals(MetadataValueType.ARRAY, MetadataValueType.fromIndex(9));
        assertEquals(MetadataValueType.UINT64, MetadataValueType.fromIndex(10));
        assertEquals(MetadataValueType.INT64, MetadataValueType.fromIndex(11));
        assertEquals(MetadataValueType.FLOAT64, MetadataValueType.fromIndex(12));
    }

    @Test
    public void testFromIndexInvalid() {
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> MetadataValueType.fromIndex(-1));
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> MetadataValueType.fromIndex(13));
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> MetadataValueType.fromIndex(100));
    }

    @Test
    public void testByteSizeFixedTypes() {
        assertEquals(1, MetadataValueType.UINT8.byteSize());
        assertEquals(1, MetadataValueType.INT8.byteSize());
        assertEquals(2, MetadataValueType.UINT16.byteSize());
        assertEquals(2, MetadataValueType.INT16.byteSize());
        assertEquals(4, MetadataValueType.UINT32.byteSize());
        assertEquals(4, MetadataValueType.INT32.byteSize());
        assertEquals(4, MetadataValueType.FLOAT32.byteSize());
        assertEquals(1, MetadataValueType.BOOL.byteSize());
        assertEquals(8, MetadataValueType.UINT64.byteSize());
        assertEquals(8, MetadataValueType.INT64.byteSize());
        assertEquals(8, MetadataValueType.FLOAT64.byteSize());
    }

    @Test
    public void testByteSizeVariableTypes() {
        // Variable-length types have negative byteSize indicating length prefix size
        assertEquals(-8, MetadataValueType.STRING.byteSize());
        assertEquals(-8, MetadataValueType.ARRAY.byteSize());
    }

    @Test
    public void testAllValuesPresent() {
        // Ensure all expected enum values exist
        assertEquals(13, MetadataValueType.values().length);

        // Verify we can access all by name
        assertNotNull(MetadataValueType.valueOf("UINT8"));
        assertNotNull(MetadataValueType.valueOf("INT8"));
        assertNotNull(MetadataValueType.valueOf("UINT16"));
        assertNotNull(MetadataValueType.valueOf("INT16"));
        assertNotNull(MetadataValueType.valueOf("UINT32"));
        assertNotNull(MetadataValueType.valueOf("INT32"));
        assertNotNull(MetadataValueType.valueOf("FLOAT32"));
        assertNotNull(MetadataValueType.valueOf("BOOL"));
        assertNotNull(MetadataValueType.valueOf("STRING"));
        assertNotNull(MetadataValueType.valueOf("ARRAY"));
        assertNotNull(MetadataValueType.valueOf("UINT64"));
        assertNotNull(MetadataValueType.valueOf("INT64"));
        assertNotNull(MetadataValueType.valueOf("FLOAT64"));
    }
}

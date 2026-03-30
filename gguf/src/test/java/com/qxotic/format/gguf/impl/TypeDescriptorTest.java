package com.qxotic.format.gguf.impl;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.MetadataValueType;
import org.junit.jupiter.api.Test;

public class TypeDescriptorTest {

    @Test
    public void testScalarNullType() {
        assertThrows(NullPointerException.class, () -> TypeDescriptor.scalar(null));
    }

    @Test
    public void testScalarArrayType() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeDescriptor.scalar(MetadataValueType.ARRAY));
    }

    @Test
    public void testArrayNullComponentType() {
        assertThrows(NullPointerException.class, () -> TypeDescriptor.array(null));
    }

    @Test
    public void testArrayArrayComponentType() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeDescriptor.array(MetadataValueType.ARRAY));
    }

    @Test
    public void testScalarTypes() {
        for (MetadataValueType type : MetadataValueType.values()) {
            if (type == MetadataValueType.ARRAY) {
                continue;
            }
            TypeDescriptor desc = TypeDescriptor.scalar(type);
            assertEquals(type, desc.type());
            assertNull(desc.componentType());
        }
    }

    @Test
    public void testArrayTypes() {
        for (MetadataValueType type : MetadataValueType.values()) {
            if (type == MetadataValueType.ARRAY) {
                continue;
            }
            TypeDescriptor desc = TypeDescriptor.array(type);
            assertEquals(MetadataValueType.ARRAY, desc.type());
            assertEquals(type, desc.componentType());
        }
    }

    @Test
    public void testCaching() {
        // TypeDescriptor uses caching - same instance should be returned
        TypeDescriptor desc1 = TypeDescriptor.scalar(MetadataValueType.INT32);
        TypeDescriptor desc2 = TypeDescriptor.scalar(MetadataValueType.INT32);
        assertSame(desc1, desc2);

        TypeDescriptor arr1 = TypeDescriptor.array(MetadataValueType.FLOAT32);
        TypeDescriptor arr2 = TypeDescriptor.array(MetadataValueType.FLOAT32);
        assertSame(arr1, arr2);
    }
}

package com.qxotic.format.gguf.impl;

import com.qxotic.format.gguf.MetadataValueType;

/**
 * Describes the type of a metadata value, including component type for arrays.
 *
 * <p>Type descriptors are cached and reused for efficiency.
 */
final class TypeDescriptor {
    private static final TypeDescriptor[] SCALARS =
            new TypeDescriptor[MetadataValueType.values().length];
    private static final TypeDescriptor[] ARRAYS =
            new TypeDescriptor[MetadataValueType.values().length];

    static {
        for (MetadataValueType valueType : MetadataValueType.values()) {
            if (valueType == MetadataValueType.ARRAY) {
                continue;
            }
            SCALARS[valueType.ordinal()] = new TypeDescriptor(valueType, null);
            ARRAYS[valueType.ordinal()] = new TypeDescriptor(MetadataValueType.ARRAY, valueType);
        }
    }

    static TypeDescriptor scalar(MetadataValueType valueType) {
        if (valueType == null) {
            throw new NullPointerException("valueType");
        }
        if (valueType == MetadataValueType.ARRAY) {
            throw new IllegalArgumentException("ARRAY requires a component type");
        }
        return SCALARS[valueType.ordinal()];
    }

    static TypeDescriptor array(MetadataValueType componentType) {
        if (componentType == null) {
            throw new NullPointerException("componentType");
        }
        if (componentType == MetadataValueType.ARRAY) {
            throw new IllegalArgumentException("Nested arrays are not supported");
        }
        return ARRAYS[componentType.ordinal()];
    }

    private final MetadataValueType type;
    private final MetadataValueType componentType;

    private TypeDescriptor(MetadataValueType type, MetadataValueType componentType) {
        this.type = type;
        this.componentType = componentType;
    }

    MetadataValueType type() {
        return type;
    }

    MetadataValueType componentType() {
        return componentType;
    }
}

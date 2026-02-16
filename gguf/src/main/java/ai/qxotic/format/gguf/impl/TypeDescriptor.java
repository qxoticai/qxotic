package ai.qxotic.format.gguf.impl;

import ai.qxotic.format.gguf.MetadataValueType;

final class TypeDescriptor {
    private static final MetadataValueType[] VALUES = MetadataValueType.values();
    private static final TypeDescriptor[] SCALARS = new TypeDescriptor[VALUES.length];
    private static final TypeDescriptor[] ARRAYS = new TypeDescriptor[VALUES.length];

    static {
        for (MetadataValueType valueType : VALUES) {
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

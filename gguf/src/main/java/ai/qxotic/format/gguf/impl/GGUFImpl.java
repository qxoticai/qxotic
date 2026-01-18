package ai.qxotic.format.gguf.impl;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.format.gguf.MetadataValueType;
import ai.qxotic.format.gguf.TensorEntry;
import java.util.*;

final class GGUFImpl implements GGUF {
    private final int version;
    private final long tensorDataOffset;
    private final Map<String, Object> metadata;
    private final Map<String, MetadataValueType> metadataTypes;
    private final Map<String, TensorEntry> tensorInfos;

    GGUFImpl(
            int version,
            long tensorDataOffset,
            Map<String, Object> metadata,
            Map<String, MetadataValueType> metadataTypes,
            Map<String, TensorEntry> tensorInfos) {
        this.version = version;
        this.tensorDataOffset = tensorDataOffset;
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        this.metadataTypes = Collections.unmodifiableMap(new LinkedHashMap<>(metadataTypes));
        this.tensorInfos = Collections.unmodifiableMap(new LinkedHashMap<>(tensorInfos));
    }

    @Override
    public int getVersion() {
        return this.version;
    }

    @Override
    public long getTensorDataOffset() {
        return this.tensorDataOffset;
    }

    @Override
    public Set<String> getMetadataKeys() {
        return this.metadata.keySet();
    }

    static <T> Class<?> toBoxedClass(Class<T> primitiveClass) {
        if (primitiveClass == boolean.class) return Boolean.class;
        if (primitiveClass == byte.class) return Byte.class;
        if (primitiveClass == char.class) return Character.class;
        if (primitiveClass == short.class) return Short.class;
        if (primitiveClass == int.class) return Integer.class;
        if (primitiveClass == long.class) return Long.class;
        if (primitiveClass == float.class) return Float.class;
        if (primitiveClass == double.class) return Double.class;
        if (primitiveClass == void.class) return Void.class;
        throw new IllegalArgumentException("not a primitive class " + primitiveClass);
    }

    @SuppressWarnings("unchecked")
    @Override
    public <T> T getValue(Class<T> targetClass, String key) {
        Object value = this.metadata.get(key);
        if (value == null) {
            // value not found
            return null;
        }
        if (targetClass.isPrimitive()) {
            return (T) toBoxedClass(targetClass).cast(value);
        } else {
            return targetClass.cast(value);
        }
    }

    @Override
    public MetadataValueType getType(String key) {
        return this.metadataTypes.get(key);
    }

    @Override
    public MetadataValueType getComponentType(String key) {
        if (!this.metadata.containsKey(key)) {
            return null;
        }
        return this.metadataTypes.get(key + "[]");
    }

    @Override
    public TensorEntry getTensor(String tensorName) {
        return this.tensorInfos.get(tensorName);
    }

    @Override
    public Collection<TensorEntry> getTensors() {
        return this.tensorInfos.values();
    }

    static long padding(long position, long alignment) {
        long nextAlignedPosition = (position + alignment - 1) / alignment * alignment;
        return nextAlignedPosition - position;
    }

    @Override
    public String toString() {
        return GGUFFormatter.toString(this, false, false);
    }
}

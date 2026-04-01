package com.qxotic.format.gguf.impl;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.MetadataValueType;
import com.qxotic.format.gguf.TensorEntry;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

final class BuilderImpl extends AbstractBuilder {

    private static final int DEFAULT_VERSION = 3;

    private int version = DEFAULT_VERSION;
    private Map<String, Object> metadata = new LinkedHashMap<>();
    private Map<String, TypeDescriptor> metadataTypes = new LinkedHashMap<>();
    private Map<String, TensorEntry> tensorInfos = new LinkedHashMap<>();

    BuilderImpl() {}

    static BuilderImpl fromExisting(GGUF gguf) {
        return new BuilderImpl()
                .setVersion(gguf.getVersion())
                .setMetadata(BuilderImpl.reconstructMetadata(gguf))
                .setMetadataTypes(BuilderImpl.reconstructTypes(gguf))
                .setTensors(BuilderImpl.fromCollection(gguf.getTensors()));
    }

    private static Map<String, Object> reconstructMetadata(GGUF gguf) {
        return gguf.getMetadataKeys().stream()
                .collect(
                        Collectors.toMap(
                                Function.identity(),
                                key -> gguf.getValue(Object.class, key),
                                (a, b) -> a,
                                LinkedHashMap::new));
    }

    private static Map<String, TypeDescriptor> reconstructTypes(GGUF gguf) {
        Map<String, TypeDescriptor> metadataTypes = new LinkedHashMap<>();
        for (String key : gguf.getMetadataKeys()) {
            MetadataValueType valueType = gguf.getType(key);
            assert valueType != null;
            metadataTypes.put(
                    key,
                    valueType == MetadataValueType.ARRAY
                            ? TypeDescriptor.array(gguf.getComponentType(key))
                            : TypeDescriptor.scalar(valueType));
        }
        return metadataTypes;
    }

    private static Map<String, TensorEntry> fromCollection(Collection<TensorEntry> tensors) {
        return tensors.stream()
                .collect(
                        Collectors.toMap(
                                TensorEntry::name,
                                Function.identity(),
                                (a, b) -> {
                                    throw new IllegalArgumentException("duplicated tensor names");
                                },
                                LinkedHashMap::new));
    }

    @Override
    public BuilderImpl setVersion(int newVersion) {
        this.version = newVersion;
        return this;
    }

    BuilderImpl setMetadata(Map<String, Object> newMetadata) {
        // Must preserve insertion order.
        assert newMetadata instanceof LinkedHashMap;
        this.metadata = newMetadata;
        return this;
    }

    BuilderImpl setMetadataTypes(Map<String, TypeDescriptor> newMetadataTypes) {
        this.metadataTypes = newMetadataTypes;
        return this;
    }

    @Override
    public GGUF build(boolean recomputeTensorOffsets) {
        assert this.metadata.keySet().stream().allMatch(key -> this.metadataTypes.containsKey(key));
        long freshTensorDataOffset = computeTensorDataOffset();
        Map<String, TensorEntry> freshTensorInfos =
                recomputeTensorOffsets ? computeTensorOffsets() : this.tensorInfos;
        return new GGUFImpl(
                this.version,
                freshTensorDataOffset,
                this.metadata,
                this.metadataTypes,
                freshTensorInfos);
    }

    @Override
    public BuilderImpl clone() {
        return new BuilderImpl()
                .setVersion(getVersion())
                .setMetadata(new LinkedHashMap<>(this.metadata))
                .setMetadataTypes(new LinkedHashMap<>(this.metadataTypes))
                .setTensors(new LinkedHashMap<>(this.tensorInfos));
    }

    @Override
    public int getVersion() {
        return this.version;
    }

    private static long sizeOfStringValue(String value) {
        return Long.BYTES // uint64_t len
                + (long) value.getBytes(StandardCharsets.UTF_8).length;
    }

    private long sizeOfTaggedValue(String key, Object value) {
        TypeDescriptor descriptor = this.metadataTypes.get(key);
        Objects.requireNonNull(descriptor);
        MetadataValueType valueType = descriptor.type();
        long totalSize = Integer.BYTES; // gguf_metadata_value_type: uint32_t type;
        switch (valueType) {
            case UINT8: // fall-through
            case INT8: // fall-through
            case UINT16: // fall-through
            case INT16: // fall-through
            case UINT32: // fall-through
            case INT32: // fall-through
            case FLOAT32: // fall-through
            case BOOL: // fall-through
            case UINT64: // fall-through
            case INT64: // fall-through
            case FLOAT64:
                totalSize += valueType.byteSize();
                break;
            case STRING:
                totalSize += sizeOfStringValue((String) value);
                break;
            case ARRAY:
                totalSize += sizeOfArray(key, value);
                break;
        }
        return totalSize;
    }

    private long sizeOfArray(String key, Object arrayValue) {
        assert arrayValue.getClass().isArray();
        TypeDescriptor descriptor = this.metadataTypes.get(key);
        if (descriptor == null) {
            throw new IllegalArgumentException("no metadata type for key '" + key + "'");
        }
        MetadataValueType componentType = descriptor.componentType();
        if (componentType == MetadataValueType.ARRAY) {
            throw new IllegalArgumentException(
                    "array of arrays not supported for key '" + key + "'");
        }
        if (componentType == MetadataValueType.STRING) {
            String[] stringArray = (String[]) arrayValue;
            long totalSize =
                    Integer.BYTES // gguf_metadata_value_type: uint32_t type;
                            + Long.BYTES; // uint64_t len;
            for (String s : stringArray) {
                totalSize += sizeOfStringValue(s);
            }
            return totalSize;
        }
        // Nested arrays are not supported yet.
        assert arrayValue.getClass().isArray()
                && (arrayValue.getClass().getComponentType() == String.class
                        || arrayValue.getClass().getComponentType().isPrimitive());
        return Integer.BYTES // gguf_metadata_value_type: uint32_t component_type;
                + Long.BYTES // uint64_t len;
                + Array.getLength(arrayValue)
                        * (long) componentType.byteSize(); // gguf_metadata_value_t array[len];
    }

    private static long sizeOfTensorInfo(TensorEntry tensorEntry) {
        return sizeOfStringValue(tensorEntry.name()) // gguf_string_t name
                + Integer.BYTES // uint32_t n_dimensions;
                + Long.BYTES
                        * (long) tensorEntry.shape().length // uint64_t dimensions[n_dimensions];
                + Integer.BYTES // ggmlType type
                + Long.BYTES; // uint64_t offset
    }

    private long computeTensorDataOffset() {
        long tensorDataOffset =
                Integer.BYTES // uint32_t MAGIC
                        + Integer.BYTES // uint32_t version
                        + Long.BYTES // uint64_t tensor_count
                        + Long.BYTES; // uint64_t metadata_kv_count;

        for (Map.Entry<String, Object> entry : this.metadata.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            tensorDataOffset += sizeOfStringValue(key);
            tensorDataOffset += sizeOfTaggedValue(key, value);
        }

        for (TensorEntry tensorEntry : this.tensorInfos.values()) {
            tensorDataOffset += sizeOfTensorInfo(tensorEntry);
        }

        int padding = (int) GGUFImpl.padding(tensorDataOffset, getAlignment());
        tensorDataOffset += padding;
        return tensorDataOffset;
    }

    BuilderImpl setTensors(Map<String, TensorEntry> newTensorInfos) {
        // Must preserve insertion order.
        assert newTensorInfos instanceof LinkedHashMap;
        assert newTensorInfos.entrySet().stream()
                .allMatch(e -> e.getKey().equals(e.getValue().name()));
        this.tensorInfos = newTensorInfos;
        return this;
    }

    @Override
    public BuilderImpl putTensor(TensorEntry tensorEntry) {
        this.tensorInfos.put(tensorEntry.name(), tensorEntry);
        return this;
    }

    @Override
    public BuilderImpl removeTensor(String tensorName) {
        this.tensorInfos.remove(tensorName);
        return this;
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
            return (T) GGUFImpl.toBoxedClass(targetClass).cast(value);
        } else {
            return targetClass.cast(value);
        }
    }

    @Override
    public TensorEntry getTensor(String tensorName) {
        return this.tensorInfos.get(tensorName);
    }

    @Override
    public Set<String> getMetadataKeys() {
        return Collections.unmodifiableSet(this.metadata.keySet());
    }

    @Override
    public Collection<TensorEntry> getTensors() {
        return Collections.unmodifiableCollection(this.tensorInfos.values());
    }

    @Override
    public MetadataValueType getType(String key) {
        TypeDescriptor descriptor = this.metadataTypes.get(key);
        return descriptor == null ? null : descriptor.type();
    }

    @Override
    public MetadataValueType getComponentType(String key) {
        TypeDescriptor descriptor = this.metadataTypes.get(key);
        return descriptor == null ? null : descriptor.componentType();
    }

    @Override
    public Builder removeKey(String key) {
        this.metadata.remove(key);
        this.metadataTypes.remove(key);
        return this;
    }

    private Map<String, TensorEntry> computeTensorOffsets() {
        long tensorOffset = 0;
        Map<String, TensorEntry> newTensorInfos = new LinkedHashMap<>();
        for (Map.Entry<String, TensorEntry> entry : tensorInfos.entrySet()) {
            // Add padding, tensor start must be aligned.
            tensorOffset += GGUFImpl.padding(tensorOffset, getAlignment());
            String name = entry.getKey();
            TensorEntry tensorEntry = entry.getValue();
            GGMLType ggmlType = tensorEntry.ggmlType();
            long byteSize = ggmlType.byteSizeForShape(tensorEntry.shape());
            newTensorInfos.put(
                    name, TensorEntry.create(name, tensorEntry.shape(), ggmlType, tensorOffset));
            tensorOffset += byteSize;
        }
        return newTensorInfos;
    }

    @Override
    protected BuilderImpl putValue(String key, MetadataValueType valueType, Object value) {
        Objects.requireNonNull(value);
        switch (valueType) {
            case UINT8: // fall-through
            case INT8:
                metadata.put(key, castTo(key, value, byte.class));
                break;
            case UINT16: // fall-through
            case INT16:
                metadata.put(key, castTo(key, value, short.class));
                break;
            case UINT32: // fall-through
            case INT32:
                metadata.put(key, castTo(key, value, int.class));
                break;
            case FLOAT32:
                metadata.put(key, castTo(key, value, float.class));
                break;
            case BOOL:
                metadata.put(key, castTo(key, value, boolean.class));
                break;
            case STRING:
                metadata.put(key, castTo(key, value, String.class));
                break;
            case UINT64: // fall-through
            case INT64:
                metadata.put(key, castTo(key, value, long.class));
                break;
            case FLOAT64:
                metadata.put(key, castTo(key, value, double.class));
                break;
            case ARRAY:
                throw new IllegalArgumentException("use putKeyArrayOf instead");
        }
        metadataTypes.put(key, TypeDescriptor.scalar(valueType));
        return this;
    }

    @Override
    protected BuilderImpl putArray(String key, MetadataValueType componentType, Object array) {
        Objects.requireNonNull(array);
        switch (componentType) {
            case UINT8: // fall-through
            case INT8:
                metadata.put(key, castTo(key, array, byte[].class));
                break;
            case UINT16: // fall-through
            case INT16:
                metadata.put(key, castTo(key, array, short[].class));
                break;
            case UINT32: // fall-through
            case INT32:
                metadata.put(key, castTo(key, array, int[].class));
                break;
            case FLOAT32:
                metadata.put(key, castTo(key, array, float[].class));
                break;
            case BOOL:
                metadata.put(key, castTo(key, array, boolean[].class));
                break;
            case STRING:
                metadata.put(key, castTo(key, array, String[].class));
                break;
            case UINT64: // fall-through
            case INT64:
                metadata.put(key, castTo(key, array, long[].class));
                break;
            case FLOAT64:
                metadata.put(key, castTo(key, array, double[].class));
                break;
            case ARRAY:
                throw new UnsupportedOperationException("array of arrays");
        }
        metadataTypes.put(key, TypeDescriptor.array(componentType));
        return this;
    }

    @SuppressWarnings("unchecked")
    private static <T> T castTo(String key, Object value, Class<? extends T> targetClass) {
        Objects.requireNonNull(value);
        try {
            if (targetClass.isPrimitive()) {
                return (T) GGUFImpl.toBoxedClass(targetClass).cast(value);
            } else {
                return targetClass.cast(value);
            }
        } catch (ClassCastException e) {
            throw new IllegalArgumentException(
                    "Expected value type "
                            + targetClass
                            + " but got "
                            + value.getClass()
                            + " for key '"
                            + key
                            + "'");
        }
    }
}

package com.qxotic.format.gguf.impl;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.MetadataValueType;

/**
 * Base implementation of {@link Builder} providing type-safe put methods for scalar and array
 * values.
 */
abstract class AbstractBuilder implements Builder {

    protected abstract BuilderImpl putValue(String key, MetadataValueType valueType, Object value);

    protected abstract BuilderImpl putArray(
            String key, MetadataValueType componentType, Object array);

    @Override
    public abstract Builder clone();

    @Override
    public boolean containsKey(String key) {
        return getValue(Object.class, key) != null;
    }

    @Override
    public boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    @Override
    public Builder putString(String key, String value) {
        return putValue(key, MetadataValueType.STRING, value);
    }

    @Override
    public Builder putBoolean(String key, boolean value) {
        return putValue(key, MetadataValueType.BOOL, value);
    }

    @Override
    public Builder putByte(String key, byte value) {
        return putValue(key, MetadataValueType.INT8, value);
    }

    @Override
    public Builder putUnsignedByte(String key, byte value) {
        return putValue(key, MetadataValueType.UINT8, value);
    }

    @Override
    public Builder putShort(String key, short value) {
        return putValue(key, MetadataValueType.INT16, value);
    }

    @Override
    public Builder putUnsignedShort(String key, short value) {
        return putValue(key, MetadataValueType.UINT16, value);
    }

    @Override
    public Builder putInteger(String key, int value) {
        return putValue(key, MetadataValueType.INT32, value);
    }

    @Override
    public Builder putUnsignedInteger(String key, int value) {
        return putValue(key, MetadataValueType.UINT32, value);
    }

    @Override
    public Builder putLong(String key, long value) {
        return putValue(key, MetadataValueType.INT64, value);
    }

    @Override
    public Builder putUnsignedLong(String key, long value) {
        return putValue(key, MetadataValueType.UINT64, value);
    }

    @Override
    public Builder putFloat(String key, float value) {
        return putValue(key, MetadataValueType.FLOAT32, value);
    }

    @Override
    public Builder putDouble(String key, double value) {
        return putValue(key, MetadataValueType.FLOAT64, value);
    }

    @Override
    public Builder putArrayOfBoolean(String key, boolean[] value) {
        return putArray(key, MetadataValueType.BOOL, value);
    }

    @Override
    public Builder putArrayOfString(String key, String[] value) {
        return putArray(key, MetadataValueType.STRING, value);
    }

    @Override
    public Builder putArrayOfByte(String key, byte[] value) {
        return putArray(key, MetadataValueType.INT8, value);
    }

    @Override
    public Builder putArrayOfUnsignedByte(String key, byte[] value) {
        return putArray(key, MetadataValueType.UINT8, value);
    }

    @Override
    public Builder putArrayOfShort(String key, short[] value) {
        return putArray(key, MetadataValueType.INT16, value);
    }

    @Override
    public Builder putArrayOfUnsignedShort(String key, short[] value) {
        return putArray(key, MetadataValueType.UINT16, value);
    }

    @Override
    public Builder putArrayOfInteger(String key, int[] value) {
        return putArray(key, MetadataValueType.INT32, value);
    }

    @Override
    public Builder putArrayOfUnsignedInteger(String key, int[] value) {
        return putArray(key, MetadataValueType.UINT32, value);
    }

    @Override
    public Builder putArrayOfLong(String key, long[] value) {
        return putArray(key, MetadataValueType.INT64, value);
    }

    @Override
    public Builder putArrayOfUnsignedLong(String key, long[] value) {
        return putArray(key, MetadataValueType.UINT64, value);
    }

    @Override
    public Builder putArrayOfFloat(String key, float[] value) {
        return putArray(key, MetadataValueType.FLOAT32, value);
    }

    @Override
    public Builder putArrayOfDouble(String key, double[] value) {
        return putArray(key, MetadataValueType.FLOAT64, value);
    }
}

package com.qxotic.format.gguf;

/**
 * Represents the different types of metadata values that can be stored in a GGUF file.
 *
 * <p>Each type has an associated byte size, which can be either positive for fixed-size types or
 * negative to indicate variable-length types with a length prefix.
 */
public enum MetadataValueType {
    /** 8-bit unsigned integer. */
    UINT8(1),
    /** 8-bit signed integer. */
    INT8(1),
    /** 16-bit unsigned little-endian integer. */
    UINT16(2),
    /** 16-bit signed little-endian integer. */
    INT16(2),
    /** 32-bit unsigned little-endian integer. */
    UINT32(4),
    /** 32-bit signed little-endian integer. */
    INT32(4),
    /** 32-bit IEEE754 floating point number. */
    FLOAT32(4),
    /**
     * Boolean. 1-byte value where 0 is false and 1 is true. Anything else is invalid, and should be
     * treated as either the model being invalid or the reader being buggy.
     */
    BOOL(1),
    /** UTF-8 non-null-terminated string, with length prepended. */
    STRING(-8),
    /**
     * Array of other values, with the length and type prepended. Arrays can be nested, and the
     * length of the array is the number of elements in the array, not the number of bytes.
     */
    ARRAY(-8),
    /** 64-bit unsigned little-endian integer. */
    UINT64(8),
    /** 64-bit signed little-endian integer. */
    INT64(8),
    /** 64-bit IEEE754 floating point number. */
    FLOAT64(8);

    /**
     * The size in bytes for this value type. Positive values indicate fixed-size types. Negative
     * values indicate variable-length types with a length prefix of the absolute value size.
     */
    private final int byteSize;

    MetadataValueType(int byteSize) {
        this.byteSize = byteSize;
    }

    /** Cache of enum values to avoid creating new arrays on each call to {@link #values()}. */
    private static final MetadataValueType[] VALUES = values();

    /**
     * Returns the MetadataValueType corresponding to the specified index.
     *
     * <p>This method is more efficient than valueOf() as it uses a cached array of values.
     *
     * @throws ArrayIndexOutOfBoundsException if the index is out of range
     */
    public static MetadataValueType fromIndex(int index) {
        return VALUES[index];
    }

    /**
     * Returns the byte size of this value type.
     *
     * <p>For fixed-size types, returns the positive size in bytes; for variable-length types,
     * returns the negative size of their length prefix.
     */
    public int byteSize() {
        return byteSize;
    }
}

package ai.qxotic.format.gguf;

/**
 * Represents the different types of metadata values that can be stored in a GGUF file.
 * Each type has an associated byte size, which can be either positive for fixed-size types
 * or negative to indicate variable-length types with a length prefix.
 */
public enum MetadataValueType {
    /**
     * The value is a 8-bit unsigned integer.
     */
    UINT8(1),
    /**
     * The value is a 8-bit signed integer.
     */
    INT8(1),
    /**
     * The value is a 16-bit unsigned little-endian integer.
     */
    UINT16(2),
    /**
     * The value is a 16-bit signed little-endian integer.
     */
    INT16(2),
    /**
     * The value is a 32-bit unsigned little-endian integer.
     */
    UINT32(4),
    /**
     * The value is a 32-bit signed little-endian integer.
     */
    INT32(4),
    /**
     * The value is a 32-bit IEEE754 floating point number.
     */
    FLOAT32(4),
    /**
     * The value is a boolean.
     * 1-byte value where 0 is false and 1 is true.
     * Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
     */
    BOOL(1),
    /**
     * The value is a UTF-8 non-null-terminated string, with length prepended.
     */
    STRING(-8),
    /**
     * The value is an array of other values, with the length and type prepended.
     * Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
     */
    ARRAY(-8),
    /**
     * The value is a 64-bit unsigned little-endian integer.
     */
    UINT64(8),
    /**
     * The value is a 64-bit signed little-endian integer.
     */
    INT64(8),
    /**
     * The value is a 64-bit IEEE754 floating point number.
     */
    FLOAT64(8);

    /**
     * The size in bytes for this value type.
     * Positive values indicate fixed-size types.
     * Negative values indicate variable-length types with a length prefix of the absolute value size.
     */
    private final int byteSize;

    /**
     * Constructs a new {@link MetadataValueType} with the specified byte size.
     *
     * @param byteSize the size in bytes for this type. Positive for fixed-size types,
     *                 negative for variable-length types indicating the size of their length prefix
     */
    MetadataValueType(int byteSize) {
        this.byteSize = byteSize;
    }

    /**
     * Cache of enum values to avoid creating new arrays on each call to {@link #values()}.
     */
    private static final MetadataValueType[] VALUES = values();

    /**
     * Returns the MetadataValueType corresponding to the specified index.
     * This method is more efficient than valueOf() as it uses a cached array of values.
     *
     * @param index the index of the desired MetadataValueType
     * @return the MetadataValueType at the specified index
     * @throws ArrayIndexOutOfBoundsException if the index is out of range
     */
    public static MetadataValueType fromIndex(int index) {
        return VALUES[index];
    }

    /**
     * Returns the byte size of this value type.
     *
     * @return for fixed-size types, returns the positive size in bytes;
     * for variable-length types, returns the negative size of their length prefix
     */
    public int byteSize() {
        return byteSize;
    }
}

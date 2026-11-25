package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.QuantizedType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.Arrays;
import java.util.Objects;

/**
 * Tensor constant {@code dense<...> : tensor<...>} (spec "TensorConstant").
 *
 * <p>Elements are stored little-endian in row-major order, each element using {@link
 * ElementType#storageBytes()} bytes (i1 uses one byte holding 0 or 1, sub-byte integers and floats
 * use one byte, complex numbers store the real part then the imaginary part). A splat stores a
 * single element that stands for every element of the tensor.
 */
public final class DenseElementsAttr implements Attribute {

    private final TensorType type;
    private final byte[] data;

    /** Above this many elements MLIR prints a hexadecimal blob instead of the elements. */
    static final int HEX_ELEMENT_LIMIT = 100;

    private final boolean splat;

    /**
     * Creates a dense attribute. As in MLIR, a tensor whose elements are all equal is stored as a
     * splat (and an empty tensor as no data), so that equal values compare equal however they were
     * spelled.
     */
    public DenseElementsAttr(TensorType type, byte[] data, boolean splat) {
        this.type = Objects.requireNonNull(type, "type");
        int elementBytes = storageBytes(type);
        long expected = (long) elementBytes * (splat ? 1 : type.numElements());
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "dense data has "
                            + data.length
                            + " bytes, expected "
                            + expected
                            + " for "
                            + type);
        }
        if (type.numElements() == 0) { // a splat of an empty tensor is just the empty tensor
            this.data = new byte[0];
            this.splat = false;
        } else if (!splat && data.length >= elementBytes && allEqual(data, elementBytes)) {
            this.data = Arrays.copyOf(data, elementBytes);
            this.splat = true;
        } else {
            this.data = data.clone();
            this.splat = splat;
        }
    }

    private static boolean allEqual(byte[] data, int elementBytes) {
        for (int i = elementBytes; i < data.length; i++) {
            if (data[i] != data[i % elementBytes]) {
                return false;
            }
        }
        return true;
    }

    /** A splat of raw element bits (for complex types: real bits only, imaginary zero). */
    public static DenseElementsAttr splatBits(TensorType type, long bits) {
        DenseElementsAttr attr = new DenseElementsAttr(type, new byte[storageBytes(type)], true);
        writeLE(attr.data, 0, storageBytes(type), bits);
        return attr;
    }

    public static int storageBytes(TensorType type) {
        return type.elementType().storageBytes();
    }

    public TensorType type() {
        return type;
    }

    public boolean isSplat() {
        return splat;
    }

    /** A copy of the raw storage. */
    public byte[] rawData() {
        return data.clone();
    }

    public int elementBytes() {
        return storageBytes(type);
    }

    public long numElements() {
        return type.numElements();
    }

    private int offset(long index) {
        if (splat) {
            return 0;
        }
        Objects.checkIndex(index, numElements());
        return Math.toIntExact(index * elementBytes());
    }

    /** Raw bits of a non-complex element (zero-extended). */
    public long rawBits(long index) {
        return readLE(data, offset(index), elementBytes());
    }

    /** Raw bits of the real ({@code part=0}) or imaginary ({@code part=1}) part of a complex. */
    public long complexPartBits(long index, int part) {
        int half = elementBytes() / 2;
        return readLE(data, offset(index) + part * half, half);
    }

    /** Integer value, sign- or zero-extended according to the (storage) integer type. */
    public long getLong(long index) {
        if (type.elementType() instanceof IndexType) {
            return rawBits(index);
        }
        IntegerType it = integerType();
        long bits = rawBits(index);
        if (it.isUnsigned() || it.isBoolean() || it.bitWidth() == 64) {
            return bits;
        }
        int shift = 64 - it.bitWidth();
        return (bits << shift) >> shift;
    }

    public boolean getBoolean(long index) {
        return rawBits(index) != 0;
    }

    /** Float value (real elements only). */
    public double getDouble(long index) {
        if (type.elementType() instanceof FloatType ft) {
            return ft.format().decode(rawBits(index));
        }
        return getLong(index);
    }

    private IntegerType integerType() {
        return switch (type.elementType()) {
            case IntegerType it -> it;
            case QuantizedType qt -> qt.storageType();
            default ->
                    throw new IllegalStateException(
                            "not an integer element type: " + type.elementType());
        };
    }

    static long readLE(byte[] buf, int off, int n) {
        long v = 0;
        for (int i = n - 1; i >= 0; i--) {
            v = (v << 8) | (buf[off + i] & 0xFFL);
        }
        return v;
    }

    static void writeLE(byte[] buf, int off, int n, long v) {
        for (int i = 0; i < n; i++) {
            buf[off + i] = (byte) (v >>> (8 * i));
        }
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DenseElementsAttr other
                && splat == other.splat
                && type.equals(other.type)
                && Arrays.equals(data, other.data);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * type.hashCode() + Arrays.hashCode(data)) + Boolean.hashCode(splat);
    }

    /** Literal of one element without the {@code dense<>} wrapper. */
    public String elementLiteral(long index) {
        ElementType et = type.elementType();
        return switch (et) {
            case IntegerType it when it.isBoolean() -> Boolean.toString(getBoolean(index));
            case IntegerType it ->
                    it.isUnsigned()
                            ? Long.toUnsignedString(getLong(index))
                            : Long.toString(getLong(index));
            case QuantizedType qt ->
                    qt.storageType().isUnsigned()
                            ? Long.toUnsignedString(getLong(index))
                            : Long.toString(getLong(index));
            case IndexType ignored -> Long.toString(getLong(index));
            case FloatType ft -> FloatAttr.literal(rawBits(index), ft.format());
            case ComplexType ct ->
                    "("
                            + FloatAttr.literal(
                                    complexPartBits(index, 0), ct.elementType().format())
                            + ","
                            + FloatAttr.literal(
                                    complexPartBits(index, 1), ct.elementType().format())
                            + ")";
        };
    }

    /**
     * MLIR spelling: splats print their element, small tensors print nested lists, larger tensors
     * (more than {@value #HEX_ELEMENT_LIMIT} elements, MLIR's default {@code
     * elementsAttrHexElementLimit}) print an exact hexadecimal blob, one byte per {@code i1}.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("dense<");
        if (numElements() == 0) {
            // MLIR spells every empty tensor dense<>
        } else if (splat) {
            sb.append(elementLiteral(0));
        } else if (numElements() <= HEX_ELEMENT_LIMIT) {
            appendNested(sb, 0, 0);
        } else {
            sb.append("\"0x");
            for (byte b : data) {
                sb.append(Character.toUpperCase(Character.forDigit((b >> 4) & 0xF, 16)));
                sb.append(Character.toUpperCase(Character.forDigit(b & 0xF, 16)));
            }
            sb.append('"');
        }
        return sb.append("> : ").append(type.mlir()).toString();
    }

    private long appendNested(StringBuilder sb, int axis, long index) {
        if (axis == type.rank()) {
            sb.append(elementLiteral(index));
            return index + 1;
        }
        sb.append('[');
        for (long i = 0; i < type.dim(axis); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            index = appendNested(sb, axis + 1, index);
        }
        sb.append(']');
        return index;
    }
}

package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.Arrays;

/**
 * A statically shaped tensor value of the interpreter: the raw bits of every element, row-major,
 * exactly as the element type stores them (floats in their own format, integers in their width;
 * complex numbers keep the real and imaginary parts in two arrays).
 */
public final class Tensor {

    private final TensorType type;
    private final long[] dims;
    private final long[] bits;
    private final long[] imag; // null unless complex

    Tensor(TensorType type, long[] bits, long[] imag) {
        if (!type.hasStaticShape()) {
            throw new InterpreterException("dynamic shapes are not supported: " + type.mlir());
        }
        this.type = type;
        this.dims = type.dims();
        this.bits = bits;
        this.imag = imag;
        if (bits.length != type.numElements()) {
            throw new IllegalArgumentException(bits.length + " elements for " + type.mlir());
        }
    }

    public TensorType type() {
        return type;
    }

    public ElementType elementType() {
        return type.elementType();
    }

    public long[] dims() {
        return dims.clone();
    }

    public int rank() {
        return dims.length;
    }

    public long dim(int axis) {
        return dims[axis];
    }

    public int size() {
        return bits.length;
    }

    boolean isComplex() {
        return imag != null;
    }

    boolean isFloat() {
        return type.elementType() instanceof FloatType;
    }

    boolean isInteger() {
        return type.elementType() instanceof IntegerType;
    }

    boolean isBoolean() {
        return type.elementType() instanceof IntegerType it && it.isBoolean();
    }

    boolean isUnsigned() {
        return type.elementType() instanceof IntegerType it && (it.isUnsigned() || it.isBoolean());
    }

    int bitWidth() {
        return switch (type.elementType()) {
            case IntegerType it -> it.bitWidth();
            case FloatType ft -> ft.format().bitWidth();
            case ComplexType ct -> 2 * ct.elementType().format().bitWidth();
            default -> throw new InterpreterException("unsupported element type " + type.mlir());
        };
    }

    long rawBits(int i) {
        return bits[i];
    }

    long rawImag(int i) {
        return imag[i];
    }

    /** Integer value: sign-extended, or zero-extended for unsigned and boolean types. */
    long getLong(int i) {
        if (!(type.elementType() instanceof IntegerType it)) {
            throw new InterpreterException("not an integer tensor: " + type.mlir());
        }
        return extend(bits[i], it);
    }

    static long extend(long raw, IntegerType it) {
        int w = it.bitWidth();
        if (w == 64) {
            return raw;
        }
        if (it.isUnsigned() || it.isBoolean()) {
            return raw & ((1L << w) - 1);
        }
        return (raw << (64 - w)) >> (64 - w);
    }

    /** Real value of a float element (or of the real part of a complex one). */
    double getDouble(int i) {
        return switch (type.elementType()) {
            case FloatType ft -> ft.format().decode(bits[i]);
            case ComplexType ct -> ct.elementType().format().decode(bits[i]);
            case IntegerType it ->
                    isUnsigned() && it.bitWidth() == 64 && bits[i] < 0
                            ? unsignedToDouble(bits[i])
                            : getLong(i);
            default -> throw new InterpreterException("unsupported element type " + type.mlir());
        };
    }

    double getImag(int i) {
        return ((ComplexType) type.elementType()).elementType().format().decode(imag[i]);
    }

    static double unsignedToDouble(long v) {
        double d = (double) (v >>> 1) * 2.0;
        return d + (v & 1);
    }

    // ================================================================== construction

    static Tensor ofBits(TensorType type, long[] bits) {
        return new Tensor(type, bits, null);
    }

    static Tensor ofComplexBits(TensorType type, long[] re, long[] im) {
        return new Tensor(type, re, im);
    }

    /** Integers wrapped to the element width. */
    static Tensor ofLongs(TensorType type, long[] values) {
        IntegerType it = (IntegerType) type.elementType();
        long[] bits = new long[values.length];
        long mask = it.bitWidth() == 64 ? -1L : (1L << it.bitWidth()) - 1;
        for (int i = 0; i < values.length; i++) {
            bits[i] = values[i] & mask;
        }
        return new Tensor(type, bits, null);
    }

    /** Floats rounded (to nearest even) to the element format. */
    static Tensor ofDoubles(TensorType type, double[] values) {
        FloatFormat f = ((FloatType) type.elementType()).format();
        long[] bits = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = f.encode(values[i]);
        }
        return new Tensor(type, bits, null);
    }

    static Tensor ofComplex(TensorType type, double[] re, double[] im) {
        FloatFormat f = ((ComplexType) type.elementType()).elementType().format();
        long[] r = new long[re.length];
        long[] m = new long[im.length];
        for (int i = 0; i < re.length; i++) {
            r[i] = f.encode(re[i]);
            m[i] = f.encode(im[i]);
        }
        return new Tensor(type, r, m);
    }

    /** {@code type} filled with element {@code i} of this tensor. */
    Tensor broadcastElement(int i, TensorType type) {
        long[] b = new long[(int) type.numElements()];
        Arrays.fill(b, bits[i]);
        long[] m = null;
        if (imag != null) {
            m = new long[b.length];
            Arrays.fill(m, imag[i]);
        }
        return new Tensor(type, b, m);
    }

    /** The elements of this tensor at {@code indices}, as a tensor of {@code type}. */
    Tensor select(TensorType type, int[] indices) {
        long[] b = new long[indices.length];
        long[] m = imag == null ? null : new long[indices.length];
        for (int k = 0; k < indices.length; k++) {
            b[k] = bits[indices[k]];
            if (m != null) {
                m[k] = imag[indices[k]];
            }
        }
        return new Tensor(type, b, m);
    }

    Tensor withType(TensorType newType) {
        return new Tensor(newType, bits, imag);
    }

    /** A rank-0 tensor holding element {@code i}. */
    Tensor scalar(int i) {
        return select(type.withShape(), new int[] {i});
    }

    long[] bitsCopy() {
        return bits.clone();
    }

    long[] imagCopy() {
        return imag == null ? null : imag.clone();
    }

    // ================================================================== indexing

    /** Row-major strides of {@code dims}. */
    static long[] strides(long[] dims) {
        long[] s = new long[dims.length];
        long acc = 1;
        for (int d = dims.length - 1; d >= 0; d--) {
            s[d] = acc;
            acc *= dims[d];
        }
        return s;
    }

    /** The multi-index of linear index {@code linear} in {@code dims}. */
    static long[] unravel(long linear, long[] dims) {
        long[] idx = new long[dims.length];
        for (int d = dims.length - 1; d >= 0; d--) {
            if (dims[d] == 0) {
                return idx;
            }
            idx[d] = linear % dims[d];
            linear /= dims[d];
        }
        return idx;
    }

    /** Linear index of {@code index} in {@code dims}, or -1 when out of bounds. */
    static long ravel(long[] index, long[] dims) {
        long linear = 0;
        for (int d = 0; d < dims.length; d++) {
            if (index[d] < 0 || index[d] >= dims[d]) {
                return -1;
            }
            linear = linear * dims[d] + index[d];
        }
        return linear;
    }

    // ================================================================== conversion

    public static Tensor of(DenseElementsAttr d) {
        TensorType t = d.type();
        int n = (int) t.numElements();
        long[] b = new long[n];
        if (t.elementType() instanceof ComplexType) {
            long[] m = new long[n];
            for (int i = 0; i < n; i++) {
                b[i] = d.complexPartBits(i, 0);
                m[i] = d.complexPartBits(i, 1);
            }
            return new Tensor(t, b, m);
        }
        if (!(t.elementType() instanceof IntegerType || t.elementType() instanceof FloatType)) {
            throw new InterpreterException("unsupported element type " + t.mlir());
        }
        for (int i = 0; i < n; i++) {
            b[i] = d.rawBits(i);
        }
        return new Tensor(t, b, null);
    }

    public DenseElementsAttr toDense() {
        int eb = DenseElementsAttr.storageBytes(type);
        byte[] data = new byte[eb * bits.length];
        for (int i = 0; i < bits.length; i++) {
            if (imag != null) {
                int half = eb / 2;
                writeLE(data, i * eb, half, bits[i]);
                writeLE(data, i * eb + half, half, imag[i]);
            } else {
                writeLE(data, i * eb, eb, bits[i]);
            }
        }
        return new DenseElementsAttr(type, data, false);
    }

    private static void writeLE(byte[] buf, int off, int n, long v) {
        for (int k = 0; k < n; k++) {
            buf[off + k] = (byte) (v >>> (8 * k));
        }
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Tensor t
                && type.equals(t.type)
                && Arrays.equals(bits, t.bits)
                && Arrays.equals(imag, t.imag);
    }

    @Override
    public int hashCode() {
        return 31 * type.hashCode() + Arrays.hashCode(bits);
    }

    @Override
    public String toString() {
        return toDense().toString();
    }
}

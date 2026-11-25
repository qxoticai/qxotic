package com.qxotic.jota.stablehlo.type;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import java.util.List;
import java.util.Objects;

/**
 * Ranked tensor type {@code tensor<2x?x3xf32[, encoding]>}. Dynamic dimensions are {@link
 * #DYNAMIC}; the optional encoding carries e.g. {@code #stablehlo.bounds<...>}.
 */
public record TensorType(List<Long> shape, ElementType elementType, Attribute encoding)
        implements Type {

    /** Marker for a dynamic ({@code ?}) dimension size. */
    public static final long DYNAMIC = -1;

    public TensorType {
        shape = List.copyOf(shape);
        Objects.requireNonNull(elementType, "elementType");
        for (long d : shape) {
            if (d < 0 && d != DYNAMIC) {
                throw new IllegalArgumentException("negative dimension size: " + d);
            }
        }
    }

    public static TensorType of(ElementType elementType, long... dims) {
        Long[] boxed = new Long[dims.length];
        for (int i = 0; i < dims.length; i++) {
            boxed[i] = dims[i];
        }
        return new TensorType(List.of(boxed), elementType, null);
    }

    public int rank() {
        return shape.size();
    }

    public long dim(int axis) {
        return shape.get(axis);
    }

    public boolean hasStaticShape() {
        return !shape.contains(DYNAMIC);
    }

    /** Number of elements; requires a static shape. */
    public long numElements() {
        long n = 1;
        for (long d : shape) {
            if (d == DYNAMIC) {
                throw new IllegalStateException("dynamic shape has no element count: " + this);
            }
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    public long[] dims() {
        long[] dims = new long[shape.size()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape.get(i);
        }
        return dims;
    }

    /** Same element type and encoding, new dimensions. */
    public TensorType withShape(long... dims) {
        Long[] boxed = new Long[dims.length];
        for (int i = 0; i < dims.length; i++) {
            boxed[i] = dims[i];
        }
        return new TensorType(List.of(boxed), elementType, encoding);
    }

    public TensorType withElementType(ElementType newElementType) {
        return new TensorType(shape, newElementType, encoding);
    }

    /**
     * Dimension bounds of a {@code #stablehlo.bounds<...>} encoding, {@link #DYNAMIC} where a
     * dimension has no bound; null when the type has no bounds encoding.
     */
    public long[] bounds() {
        if (!(encoding instanceof DialectAttr d)
                || !d.dialect().equals("stablehlo")
                || !d.mnemonic().equals("bounds")
                || d.rawBody() == null) {
            return null;
        }
        String body = d.rawBody().trim();
        if (body.isEmpty()) {
            return new long[0];
        }
        String[] parts = body.split(",");
        long[] out = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String p = parts[i].trim();
            out[i] = p.equals("?") || p.startsWith("-") ? DYNAMIC : Long.parseLong(p);
        }
        return out;
    }

    /**
     * The {@code #stablehlo.bounds<...>} encoding of {@code bounds}, or null when every bound is
     * {@link #DYNAMIC}.
     */
    public static Attribute boundsEncoding(long[] bounds) {
        boolean any = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bounds.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(bounds[i] == DYNAMIC ? "?" : Long.toString(bounds[i]));
            any |= bounds[i] != DYNAMIC;
        }
        return any ? DialectAttr.ofRaw("stablehlo", "bounds", sb.toString()) : null;
    }

    @Override
    public String mlir() {
        StringBuilder sb = new StringBuilder("tensor<");
        for (long d : shape) {
            sb.append(d == DYNAMIC ? "?" : Long.toString(d)).append('x');
        }
        sb.append(elementType.mlir());
        if (encoding != null) {
            sb.append(", ").append(encoding);
        }
        return sb.append('>').toString();
    }

    @Override
    public String toString() {
        return mlir();
    }
}

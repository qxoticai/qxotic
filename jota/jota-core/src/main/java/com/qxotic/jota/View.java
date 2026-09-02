package com.qxotic.jota;

/**
 * A typed addressing recipe over storage: {@link #layout()}, {@link #dataType()} and a base {@link
 * #byteOffset()}. Immutable and storage-free; the storage unit at coordinate {@code c} starts at
 * {@code byteOffset() + Σ c[i] × stride[i] × dataType().byteSize()}.
 */
public interface View {
    Layout layout();

    DataType dataType();

    /** The absolute base offset, in bytes, that the layout's offsets are relative to. */
    long byteOffset();

    /**
     * The PHYSICAL (storage-unit) shape of this view: for block-quantized dtypes the innermost axis
     * counts blocks, not elements - see {@link DataType}. All view algebra (slice, reshape,
     * transpose) operates on physical units; blocks are atomic. {@link #logicalShape()} is the
     * element-dimensioned read-back.
     */
    default Shape shape() {
        return layout().shape();
    }

    /**
     * The element-dimensioned read-back of {@link #shape()}: identical except for block-quantized
     * dtypes, where the innermost axis is multiplied by {@code elementsPerBlock()} (see {@link
     * DataType#logicalShape(Shape)}). A convenience for code that thinks in elements (config
     * cross-checks, output dims) - it never authorizes addressing: all storage math stays physical.
     * Defined on the view's storage axis order (blocked axis last); for a permuted view consult
     * {@link #layout()} directly.
     */
    default Shape logicalShape() {
        return dataType().logicalShape(shape());
    }

    /** {@code logicalShape().size()} - the element count. */
    default long logicalSize() {
        return logicalShape().size();
    }

    default Stride stride() {
        return layout().stride();
    }

    /** {@link #stride()} scaled to bytes ({@code × dataType().byteSize()}). */
    default Stride byteStride() {
        return stride().scale(dataType().byteSize());
    }
}

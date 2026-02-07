package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;

/**
 * View transform operation in IR-T. Represents operations that only change the layout (shape +
 * stride) without allocating new memory.
 *
 * <p>The {@code kind} describes the transformation type with its parameters, enabling lazy index
 * computation at LIR lowering time for complex cases (e.g., transpose followed by reshape).
 *
 * <p>When {@code needsLazyIndexing} is true, the strides in {@code layout} are placeholders and the
 * actual index computation must be performed by walking the ViewTransform chain at lowering time.
 */
public record ViewTransform(TIRNode input, ViewKind kind, Layout layout, boolean needsLazyIndexing)
        implements TIRNode {

    public ViewTransform {
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        if (kind == null) {
            throw new IllegalArgumentException("kind cannot be null");
        }
        if (layout == null) {
            throw new IllegalArgumentException("layout cannot be null");
        }
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Shape shape() {
        return layout.shape();
    }

    /** Returns a hint string for debugging/display (derived from kind). */
    public String hint() {
        return switch (kind) {
            case ViewKind.Transpose t -> "transpose";
            case ViewKind.Reshape r -> "view";
            case ViewKind.Broadcast b -> "broadcast";
            case ViewKind.Expand e -> "expand";
            case ViewKind.Slice s -> "slice";
        };
    }
}

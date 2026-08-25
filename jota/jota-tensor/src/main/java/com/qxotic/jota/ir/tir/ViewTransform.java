package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

/**
 * View transform operation in IR-T. Represents operations that only change the layout (shape +
 * stride) without allocating new memory.
 *
 * <p>The {@code operation} carries the parameters needed to compute indices while lowering complex
 * view chains.
 *
 * <p>When {@code needsLazyIndexing} is true, the strides in {@code layout} are placeholders and the
 * actual index computation must be performed by walking the ViewTransform chain at lowering time.
 */
public record ViewTransform(
        TIRNode input, ViewOperation operation, Layout layout, boolean needsLazyIndexing)
        implements TIRNode {

    public ViewTransform {
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        if (operation == null) {
            throw new IllegalArgumentException("operation cannot be null");
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

    /** Returns a hint string for debugging and display. */
    public String hint() {
        return switch (operation) {
            case ViewOperation.Transpose __ -> "transpose";
            case ViewOperation.Reshape __ -> "view";
            case ViewOperation.Broadcast __ -> "broadcast";
            case ViewOperation.Expand __ -> "expand";
            case ViewOperation.Slice __ -> "slice";
        };
    }
}

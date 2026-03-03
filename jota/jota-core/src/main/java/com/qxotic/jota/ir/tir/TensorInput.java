package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

/**
 * Tensor input node in IR-T. Represents an input tensor to the IR graph, already materialized with
 * a specific layout.
 */
public record TensorInput(int id, DataType dataType, Layout layout) implements TIRNode {

    public TensorInput {
        if (id < 0) {
            throw new IllegalArgumentException("TensorInput id must be non-negative, got: " + id);
        }
    }

    @Override
    public Shape shape() {
        return layout.shape();
    }
}

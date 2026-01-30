package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;

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
}

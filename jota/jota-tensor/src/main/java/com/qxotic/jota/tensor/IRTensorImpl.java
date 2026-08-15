package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.ir.tir.ViewTransform;
import com.qxotic.jota.memory.MemoryView;
import java.util.Objects;
import java.util.Optional;

/**
 * Temporary wrapper for TIRNode during IR-T tracing. Similar to TraceTensor but uses IR-T instead
 * of ExprNode.
 */
final class IRTensorImpl extends AbstractTensorImpl {

    private final TIRNode node;
    private final Device device;

    IRTensorImpl(TIRNode node, Device device) {
        this.node = Objects.requireNonNull(node);
        this.device = Objects.requireNonNull(device);
    }

    @Override
    public DataType dataType() {
        return node.dataType();
    }

    @Override
    public Layout layout() {
        if (node instanceof TensorInput input) {
            return input.layout();
        }
        if (node instanceof ViewTransform view) {
            return view.layout();
        }
        return Layout.rowMajor(node.shape());
    }

    @Override
    public Device device() {
        return device;
    }

    boolean isMaterializedInternal() {
        return false;
    }

    boolean isLazyInternal() {
        return true;
    }

    Optional<MemoryView<?>> tryGetMaterializedInternal() {
        return Optional.empty();
    }

    @Override
    public MemoryView<?> materialize() {
        throw new UnsupportedOperationException(
                "IRTensor cannot be materialized directly. "
                        + "Use tracing to build a lazy tensor graph.");
    }

    Optional<LazyComputation> computationInternal() {
        return Optional.empty();
    }

    TIRNode node() {
        return node;
    }
}

package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Objects;
import java.util.Optional;

/**
 * Temporary wrapper for TIRNode during IR-T tracing. Similar to TraceTensor but uses IR-T instead
 * of ExprNode.
 */
final class IRTensor implements Tensor {

    private final TIRNode node;
    private final Device device;

    IRTensor(TIRNode node, Device device) {
        this.node = Objects.requireNonNull(node);
        this.device = Objects.requireNonNull(device);
    }

    @Override
    public DataType dataType() {
        return node.dataType();
    }

    @Override
    public Layout layout() {
        return node.layout();
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public boolean isMaterialized() {
        return false;
    }

    @Override
    public boolean isLazy() {
        return true;
    }

    @Override
    public Optional<MemoryView<?>> tryGetMaterialized() {
        return Optional.empty();
    }

    @Override
    public MemoryView<?> materialize() {
        throw new UnsupportedOperationException(
                "IRTensor cannot be materialized directly. "
                        + "Use Tensor.lazy(IRComputation) or trace to IRGraph.");
    }

    @Override
    public Optional<LazyComputation> computation() {
        return Optional.empty();
    }

    TIRNode node() {
        return node;
    }
}

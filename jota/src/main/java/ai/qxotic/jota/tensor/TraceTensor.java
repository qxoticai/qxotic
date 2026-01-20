package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Optional;

final class TraceTensor implements Tensor {

    private final ExprNode node;

    TraceTensor(ExprNode node) {
        this.node = node;
    }

    ExprNode node() {
        return node;
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
        return node.device();
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
    public MemoryView<?> materialize() {
        throw new UnsupportedOperationException("Trace tensors cannot be materialized directly");
    }

    @Override
    public Optional<MemoryView<?>> tryGetMaterialized() {
        return Optional.empty();
    }

    @Override
    public Optional<LazyComputation> computation() {
        return Optional.empty();
    }
}

package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Objects;
import java.util.Optional;

final class MaterializedTensor implements Tensor {

    private final MemoryView<?> view;

    MaterializedTensor(MemoryView<?> view) {
        this.view = Objects.requireNonNull(view);
    }

    @Override
    public DataType dataType() {
        return view.dataType();
    }

    @Override
    public Layout layout() {
        return view.layout();
    }

    @Override
    public Device device() {
        return view.memory().device();
    }

    @Override
    public boolean isMaterialized() {
        return true;
    }

    @Override
    public boolean isLazy() {
        return false;
    }

    @Override
    public MemoryView<?> materialize() {
        return view;
    }

    @Override
    public Optional<MemoryView<?>> tryGetMaterialized() {
        return Optional.of(view);
    }

    @Override
    public Optional<LazyComputation> computation() {
        return Optional.empty();
    }
}

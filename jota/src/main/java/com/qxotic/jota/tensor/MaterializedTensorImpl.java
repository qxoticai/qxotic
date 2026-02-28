package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.memory.MemoryView;
import java.util.Objects;
import java.util.Optional;

final class MaterializedTensorImpl extends AbstractTensorImpl {

    private final MemoryView<?> view;

    MaterializedTensorImpl(MemoryView<?> view) {
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

    boolean isMaterializedInternal() {
        return true;
    }

    boolean isLazyInternal() {
        return false;
    }

    @Override
    public MemoryView<?> materialize() {
        return view;
    }

    Optional<MemoryView<?>> tryGetMaterializedInternal() {
        return Optional.of(view);
    }

    Optional<LazyComputation> computationInternal() {
        return Optional.empty();
    }

    @Override
    public String toString() {
        return "Tensor(materialized=true, lazy=false, dtype="
                + dataType()
                + ", shape="
                + layout().shape()
                + ", layout="
                + layout()
                + ", device="
                + device().name()
                + ")";
    }
}

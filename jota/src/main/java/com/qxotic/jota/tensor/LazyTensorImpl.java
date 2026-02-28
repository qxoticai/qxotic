package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.memory.MemoryView;
import java.util.Objects;
import java.util.Optional;

final class LazyTensorImpl extends AbstractTensorImpl {

    private final LazyComputation computation;
    private final DataType dtype;
    private final Layout layout;
    private final Device device;

    private volatile MemoryView<?> cachedResult;
    private final Object materializeLock = new Object();

    LazyTensorImpl(LazyComputation computation, DataType dtype, Layout layout, Device device) {
        this.computation = Objects.requireNonNull(computation);
        this.dtype = Objects.requireNonNull(dtype);
        this.layout = Objects.requireNonNull(layout);
        this.device = Objects.requireNonNull(device);
    }

    @Override
    public DataType dataType() {
        return dtype;
    }

    @Override
    public Layout layout() {
        return layout;
    }

    @Override
    public Device device() {
        return device;
    }

    boolean isMaterializedInternal() {
        return cachedResult != null;
    }

    boolean isLazyInternal() {
        return true;
    }

    @Override
    public MemoryView<?> materialize() {
        if (cachedResult != null) {
            return cachedResult;
        }

        synchronized (materializeLock) {
            if (cachedResult != null) {
                return cachedResult;
            }

            cachedResult = computation.execute();
            return cachedResult;
        }
    }

    Optional<MemoryView<?>> tryGetMaterializedInternal() {
        return Optional.ofNullable(cachedResult);
    }

    Optional<LazyComputation> computationInternal() {
        return Optional.of(computation);
    }

    @Override
    public String toString() {
        StringBuilder builder =
                new StringBuilder("Tensor(materialized=")
                        .append(isMaterializedInternal())
                        .append(", lazy=true, dtype=")
                        .append(dtype)
                        .append(", shape=")
                        .append(layout.shape())
                        .append(", layout=")
                        .append(layout)
                        .append(", device=")
                        .append(device.name());
        String computationType = computation.getClass().getSimpleName();
        if ("ConstantComputation".equals(computationType)) {
            builder.append(", op=constant");
        } else if ("RangeComputation".equals(computationType)) {
            builder.append(", op=range");
        } else if ("RandomComputation".equals(computationType)) {
            builder.append(", op=random");
        } else {
            builder.append(", op=ir-graph");
        }
        builder.append(")");
        return builder.toString();
    }
}

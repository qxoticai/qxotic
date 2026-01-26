package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;

public final class ConstantView implements View {
    private final Immediate storage;
    private final Layout layout;
    private final DataType dataType;

    private ConstantView(Immediate storage, Shape shape, DataType dataType) {
        if (storage.dataType() != dataType) {
            throw new IllegalArgumentException(
                    "storage data type mismatch: " + storage.dataType() + " vs " + dataType);
        }
        this.storage = storage;
        this.layout = Layout.of(shape, Stride.zeros(shape));
        this.dataType = dataType;
    }

    public static ConstantView of(Immediate storage, Shape shape, DataType dataType) {
        return new ConstantView(storage, shape, dataType);
    }

    @Override
    public Immediate storage() {
        return storage;
    }

    @Override
    public Layout layout() {
        return layout;
    }

    @Override
    public DataType dataType() {
        return dataType;
    }
}

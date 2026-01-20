package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

sealed interface ValueSpec permits TensorSpec, ScalarSpec {}

record TensorSpec(DataType type, Layout layout, Device device) implements ValueSpec {
    static TensorSpec from(Tensor tensor) {
        return new TensorSpec(tensor.dataType(), tensor.layout(), tensor.device());
    }
}

record ScalarSpec(DataType type) implements ValueSpec {}

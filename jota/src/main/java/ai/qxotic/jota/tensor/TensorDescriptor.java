package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;

public record TensorDescriptor(String name, int rank, DataType dtype, Layout layout) {

    public static TensorDescriptor any(String name) {
        return new TensorDescriptor(name, -1, null, null);
    }

    public static TensorDescriptor matrix(String name, DataType dtype) {
        return new TensorDescriptor(name, 2, dtype, null);
    }
}

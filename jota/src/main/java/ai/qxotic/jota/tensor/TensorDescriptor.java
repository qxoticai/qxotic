package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;

public record TensorDescriptor(
        String name, KernelInputKind kind, int rank, DataType dtype, Layout layout) {

    public static TensorDescriptor any(String name) {
        return new TensorDescriptor(name, KernelInputKind.TENSOR, -1, null, null);
    }

    public static TensorDescriptor matrix(String name, DataType dtype) {
        return new TensorDescriptor(name, KernelInputKind.TENSOR, 2, dtype, null);
    }

    public static TensorDescriptor scalar(String name, DataType dtype) {
        return new TensorDescriptor(name, KernelInputKind.SCALAR, -1, dtype, null);
    }
}

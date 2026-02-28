package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.memory.MemoryView;
import java.util.Optional;
import java.util.OptionalLong;

public final class TensorTestInternals {

    private TensorTestInternals() {}

    public static boolean isMaterialized(Tensor tensor) {
        return InternalTensorAccess.isMaterialized(tensor);
    }

    public static boolean isLazy(Tensor tensor) {
        return InternalTensorAccess.isLazy(tensor);
    }

    public static Optional<MemoryView<?>> tryGetMaterialized(Tensor tensor) {
        return InternalTensorAccess.tryGetMaterialized(tensor);
    }

    public static Optional<LazyComputation> computation(Tensor tensor) {
        return InternalTensorAccess.computation(tensor);
    }

    public static boolean isScalarBroadcast(Tensor tensor) {
        return InternalTensorAccess.isScalarBroadcast(tensor);
    }

    public static OptionalLong scalarConstantBits(Tensor tensor) {
        return InternalTensorAccess.scalarConstantBits(tensor);
    }

    public static Tensor createMaterialized(MemoryView<?> view) {
        return new MaterializedTensorImpl(view);
    }

    public static Tensor createLazy(
            LazyComputation computation, DataType dtype, Layout layout, Device device) {
        return new LazyTensorImpl(computation, dtype, layout, device);
    }

    public static Tensor createIRTensor(TIRNode node, Device device) {
        return new IRTensorImpl(node, device);
    }
}

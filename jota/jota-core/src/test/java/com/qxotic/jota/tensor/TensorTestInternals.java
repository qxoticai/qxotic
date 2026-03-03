package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.memory.MemoryView;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;

public final class TensorTestInternals {

    private TensorTestInternals() {}

    public static boolean isMaterialized(Tensor tensor) {
        return requireImpl(tensor).isMaterializedInternal();
    }

    public static boolean isLazy(Tensor tensor) {
        return requireImpl(tensor).isLazyInternal();
    }

    public static Optional<MemoryView<?>> tryGetMaterialized(Tensor tensor) {
        return requireImpl(tensor).tryGetMaterializedInternal();
    }

    public static Optional<Object> computation(Tensor tensor) {
        return InternalTensorAccess.computation(tensor).map(c -> (Object) c);
    }

    public static Optional<Map<String, Object>> computationAttributes(Tensor tensor) {
        return InternalTensorAccess.computation(tensor).map(LazyComputation::attributes);
    }

    public static boolean isScalarBroadcast(Tensor tensor) {
        return requireImpl(tensor).isScalarBroadcastInternal();
    }

    public static OptionalLong scalarConstantBits(Tensor tensor) {
        return requireImpl(tensor).scalarConstantBitsInternal();
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

    private static AbstractTensorImpl requireImpl(Tensor tensor) {
        if (tensor instanceof AbstractTensorImpl impl) {
            return impl;
        }
        throw new IllegalArgumentException(
                "Unsupported Tensor implementation for test internals: " + tensor.getClass());
    }
}

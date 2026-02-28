package com.qxotic.jota.tensor;

import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.memory.MemoryView;
import java.util.Optional;
import java.util.OptionalLong;

final class InternalTensorAccess {

    private InternalTensorAccess() {}

    static boolean isMaterialized(Tensor tensor) {
        return requireImpl(tensor).isMaterializedInternal();
    }

    static boolean isLazy(Tensor tensor) {
        return requireImpl(tensor).isLazyInternal();
    }

    static Optional<MemoryView<?>> tryGetMaterialized(Tensor tensor) {
        return requireImpl(tensor).tryGetMaterializedInternal();
    }

    static Optional<LazyComputation> computation(Tensor tensor) {
        return requireImpl(tensor).computationInternal();
    }

    static boolean isScalarBroadcast(Tensor tensor) {
        return requireImpl(tensor).isScalarBroadcastInternal();
    }

    static OptionalLong scalarConstantBits(Tensor tensor) {
        return requireImpl(tensor).scalarConstantBitsInternal();
    }

    static TIRNode irNode(Tensor tensor) {
        if (tensor instanceof IRTensorImpl irTensor) {
            return irTensor.node();
        }
        throw new IllegalArgumentException("Expected IRTensorImpl, got: " + tensor.getClass());
    }

    private static AbstractTensorImpl requireImpl(Tensor tensor) {
        if (tensor instanceof AbstractTensorImpl impl) {
            return impl;
        }
        throw new IllegalArgumentException(
                "Unsupported Tensor implementation for internal access: " + tensor.getClass());
    }
}

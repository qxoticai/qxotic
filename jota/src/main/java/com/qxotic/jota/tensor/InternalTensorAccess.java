package com.qxotic.jota.tensor;

import com.qxotic.jota.ir.tir.TIRNode;
import java.util.Optional;

final class InternalTensorAccess {

    private InternalTensorAccess() {}

    static boolean isMaterialized(Tensor tensor) {
        return requireImpl(tensor).isMaterializedInternal();
    }

    static Optional<LazyComputation> computation(Tensor tensor) {
        return requireImpl(tensor).computationInternal();
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

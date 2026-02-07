package ai.qxotic.jota.tensor;

import java.util.List;

final class KernelValidation {

    private KernelValidation() {}

    static boolean matches(KernelSignature signature, KernelInput input) {
        if (signature == null) {
            return true;
        }
        List<TensorDescriptor> descriptors = signature.inputs();
        if (descriptors != null && descriptors.size() != input.size()) {
            return false;
        }
        if (descriptors != null) {
            for (int i = 0; i < descriptors.size(); i++) {
                TensorDescriptor desc = descriptors.get(i);
                KernelInputEntry entry = input.entry(i);
                if (desc.kind() == KernelInputKind.SCALAR) {
                    if (entry.kind() == KernelInputKind.SCALAR) {
                        if (desc.dtype() != null && entry.dataType() != desc.dtype()) {
                            return false;
                        }
                    } else if (entry.kind() == KernelInputKind.TENSOR) {
                        Tensor tensor = (Tensor) entry.value();
                        boolean isConstantScalar =
                                tensor.isScalar()
                                        && tensor.computation()
                                                .filter(ConstantComputation.class::isInstance)
                                                .isPresent();
                        if (!isConstantScalar) {
                            return false;
                        }
                        if (desc.dtype() != null && tensor.dataType() != desc.dtype()) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                    continue;
                }
                if (entry.kind() != KernelInputKind.TENSOR) {
                    return false;
                }
                Tensor tensor = (Tensor) entry.value();
                if (desc.rank() >= 0 && tensor.shape().rank() != desc.rank()) {
                    return false;
                }
                if (desc.dtype() != null && tensor.dataType() != desc.dtype()) {
                    return false;
                }
                if (desc.layout() != null && !tensor.layout().equals(desc.layout())) {
                    return false;
                }
            }
        }
        return true;
    }
}

package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import java.util.List;
import java.util.Optional;

public interface KernelExecutor {
    KernelOutput execute(String name, KernelInput input, ExecutionContext ctx);

    static KernelExecutor auto(KernelRegistry registry) {
        return new AutoKernelExecutor(registry);
    }

    static KernelExecutor backend(KernelRegistry registry, Device backend) {
        return new BackendKernelExecutor(registry, backend);
    }

    final class AutoKernelExecutor implements KernelExecutor {
        private final KernelRegistry registry;

        AutoKernelExecutor(KernelRegistry registry) {
            this.registry = registry;
        }

        @Override
        public KernelOutput execute(String name, KernelInput input, ExecutionContext ctx) {
            Device device = ctx.device();
            Optional<Kernel<KernelInput, KernelOutput>> kernel =
                    registry.find(name, device, input).map(k -> (Kernel<KernelInput, KernelOutput>) k);
            return kernel
                    .orElseThrow(
                            () ->
                                    new UnsupportedOperationException(
                                            "No kernel '" + name + "' for device " + device))
                    .execute(input, ctx);
        }
    }

    final class BackendKernelExecutor implements KernelExecutor {
        private final KernelRegistry registry;
        private final Device backend;

        BackendKernelExecutor(KernelRegistry registry, Device backend) {
            this.registry = registry;
            this.backend = backend;
        }

        @SuppressWarnings("unchecked")
        @Override
        public KernelOutput execute(String name, KernelInput input, ExecutionContext ctx) {
            List<Kernel<?, ?>> kernels = registry.implementations(name);
            for (Kernel<?, ?> candidate : kernels) {
                Kernel<KernelInput, KernelOutput> kernel =
                        (Kernel<KernelInput, KernelOutput>) candidate;
                if (!kernel.supports(backend)) {
                    continue;
                }
                if (!KernelValidation.matches(kernel.signature(), input)) {
                    continue;
                }
                return kernel.execute(input, ctx);
            }
            throw new UnsupportedOperationException(
                    "No kernel '" + name + "' for backend " + backend);
        }
    }
}

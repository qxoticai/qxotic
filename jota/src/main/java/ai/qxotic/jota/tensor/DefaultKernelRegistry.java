package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public final class DefaultKernelRegistry implements KernelRegistry {

    private final Map<String, List<PrioritizedKernel>> kernels = new ConcurrentHashMap<>();

    @Override
    public <I extends KernelInput, O extends KernelOutput> void register(String name, Kernel<I, O> kernel) {
        register(name, kernel, 0);
    }

    @Override
    public <I extends KernelInput, O extends KernelOutput> void register(
            String name, Kernel<I, O> kernel, int priority) {
        kernels.compute(
                name,
                (k, list) -> {
                    List<PrioritizedKernel> updated =
                            list == null ? new ArrayList<>() : new ArrayList<>(list);
                    updated.add(new PrioritizedKernel(kernel, priority));
                    updated.sort(Comparator.comparingInt(PrioritizedKernel::priority).reversed());
                    return updated;
                });
    }

    @SuppressWarnings("unchecked")
    @Override
    public <I extends KernelInput, O extends KernelOutput> Optional<Kernel<I, O>> find(
            String name, Device device, I input) {
        List<PrioritizedKernel> candidates = kernels.get(name);
        if (candidates == null) {
            return Optional.empty();
        }
        for (PrioritizedKernel candidate : candidates) {
            Kernel<I, O> kernel = (Kernel<I, O>) candidate.kernel();
            if (!kernel.supports(device)) {
                continue;
            }
            if (!KernelValidation.matches(kernel.signature(), input)) {
                continue;
            }
            return Optional.of(kernel);
        }
        return Optional.empty();
    }

    @Override
    public List<Kernel<?, ?>> implementations(String name) {
        List<PrioritizedKernel> list = kernels.get(name);
        if (list == null) {
            return List.of();
        }
        List<Kernel<?, ?>> result = new ArrayList<>();
        for (PrioritizedKernel entry : list) {
            result.add(entry.kernel());
        }
        return result;
    }

    private record PrioritizedKernel(Kernel<?, ?> kernel, int priority) {}
}

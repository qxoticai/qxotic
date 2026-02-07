package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import java.util.List;
import java.util.Optional;

public interface KernelRegistry {
    <I extends KernelInput, O extends KernelOutput> void register(String name, Kernel<I, O> kernel);

    <I extends KernelInput, O extends KernelOutput> void register(
            String name, Kernel<I, O> kernel, int priority);

    <I extends KernelInput, O extends KernelOutput> Optional<Kernel<I, O>> find(
            String name, Device device, I input);

    List<Kernel<?, ?>> implementations(String name);
}

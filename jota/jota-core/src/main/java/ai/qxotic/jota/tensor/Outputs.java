package ai.qxotic.jota.tensor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Outputs {

    private Outputs() {}

    public static KernelOutput of(Tensor... tensors) {
        return new DefaultKernelOutput(List.of(tensors), Map.of());
    }

    public static KernelOutput of(Map<String, Tensor> named) {
        return new DefaultKernelOutput(List.copyOf(named.values()), new LinkedHashMap<>(named));
    }

    private static final class DefaultKernelOutput implements KernelOutput {
        private final List<Tensor> ordered;
        private final Map<String, Tensor> named;

        private DefaultKernelOutput(List<Tensor> ordered, Map<String, Tensor> named) {
            this.ordered = List.copyOf(ordered);
            this.named = Map.copyOf(named);
        }

        @Override
        public Tensor get(int index) {
            return ordered.get(index);
        }

        @Override
        public Tensor get(String name) {
            Tensor tensor = named.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("Missing output: " + name);
            }
            return tensor;
        }

        @Override
        public int size() {
            return ordered.size();
        }
    }
}

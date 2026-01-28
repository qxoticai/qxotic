package ai.qxotic.jota.tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class Inputs {

    private Inputs() {}

    public static KernelInput of(Tensor... tensors) {
        return new DefaultKernelInput(List.of(tensors), Map.of(), Map.of());
    }

    public static KernelInput of(Map<String, Tensor> named) {
        return new DefaultKernelInput(
                List.copyOf(named.values()), new LinkedHashMap<>(named), Map.of());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final Map<String, Tensor> tensors = new LinkedHashMap<>();
        private final Map<String, Object> params = new LinkedHashMap<>();

        public Builder tensor(String name, Tensor tensor) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(tensor, "tensor");
            tensors.put(name, tensor);
            return this;
        }

        public Builder param(String name, Object value) {
            Objects.requireNonNull(name, "name");
            params.put(name, value);
            return this;
        }

        public KernelInput build() {
            return new DefaultKernelInput(new ArrayList<>(tensors.values()), tensors, params);
        }
    }

    private static final class DefaultKernelInput implements KernelInput {
        private final List<Tensor> ordered;
        private final Map<String, Tensor> named;
        private final Map<String, Object> params;

        private DefaultKernelInput(
                List<Tensor> ordered, Map<String, Tensor> named, Map<String, Object> params) {
            this.ordered = List.copyOf(ordered);
            this.named = Map.copyOf(named);
            this.params = Map.copyOf(params);
        }

        @Override
        public Tensor get(int index) {
            return ordered.get(index);
        }

        @Override
        public Tensor get(String name) {
            Tensor tensor = named.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException("Missing tensor: " + name);
            }
            return tensor;
        }

        @Override
        public int size() {
            return ordered.size();
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T param(String name, Class<T> type) {
            Object value = params.get(name);
            if (value == null) {
                throw new IllegalArgumentException("Missing param: " + name);
            }
            if (!type.isInstance(value)) {
                throw new IllegalArgumentException("Param " + name + " is not of type " + type);
            }
            return (T) value;
        }
    }
}

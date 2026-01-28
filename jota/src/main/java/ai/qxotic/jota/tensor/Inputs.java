package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class Inputs {

    private Inputs() {}

    public static KernelInput of(Tensor... tensors) {
        List<KernelInputEntry> ordered = new ArrayList<>(tensors.length);
        for (Tensor tensor : tensors) {
            ordered.add(new KernelInputEntry(KernelInputKind.TENSOR, tensor, tensor.dataType()));
        }
        return new DefaultKernelInput(ordered, Map.of());
    }

    public static KernelInput of(Map<String, Tensor> named) {
        List<KernelInputEntry> ordered = new ArrayList<>(named.size());
        Map<String, KernelInputEntry> namedEntries = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> entry : named.entrySet()) {
            KernelInputEntry inputEntry =
                    new KernelInputEntry(
                            KernelInputKind.TENSOR, entry.getValue(), entry.getValue().dataType());
            ordered.add(inputEntry);
            namedEntries.put(entry.getKey(), inputEntry);
        }
        return new DefaultKernelInput(ordered, namedEntries);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final Map<String, KernelInputEntry> namedEntries = new LinkedHashMap<>();
        private final List<KernelInputEntry> ordered = new ArrayList<>();

        public Builder tensor(String name, Tensor tensor) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(tensor, "tensor");
            KernelInputEntry entry =
                    new KernelInputEntry(KernelInputKind.TENSOR, tensor, tensor.dataType());
            namedEntries.put(name, entry);
            ordered.add(entry);
            return this;
        }

        public Builder scalar(String name, Object value) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(value, "value");
            DataType dataType = inferScalarType(value);
            KernelInputEntry entry = new KernelInputEntry(KernelInputKind.SCALAR, value, dataType);
            namedEntries.put(name, entry);
            ordered.add(entry);
            return this;
        }

        public Builder param(String name, Object value) {
            return scalar(name, value);
        }

        public KernelInput build() {
            return new DefaultKernelInput(
                    new ArrayList<>(ordered), new LinkedHashMap<>(namedEntries));
        }
    }

    private static final class DefaultKernelInput implements KernelInput {
        private final List<KernelInputEntry> ordered;
        private final Map<String, KernelInputEntry> named;

        private DefaultKernelInput(
                List<KernelInputEntry> ordered, Map<String, KernelInputEntry> named) {
            this.ordered = List.copyOf(ordered);
            this.named = Map.copyOf(named);
        }

        @Override
        public Tensor get(int index) {
            KernelInputEntry entry = ordered.get(index);
            if (entry.kind() != KernelInputKind.TENSOR) {
                throw new IllegalArgumentException("Input at index " + index + " is not a tensor");
            }
            return (Tensor) entry.value();
        }

        @Override
        public Tensor get(String name) {
            KernelInputEntry entry = named.get(name);
            if (entry == null) {
                throw new IllegalArgumentException("Missing tensor: " + name);
            }
            if (entry.kind() != KernelInputKind.TENSOR) {
                throw new IllegalArgumentException("Input " + name + " is not a tensor");
            }
            return (Tensor) entry.value();
        }

        @Override
        public KernelInputEntry entry(int index) {
            return ordered.get(index);
        }

        @Override
        public KernelInputEntry entry(String name) {
            KernelInputEntry entry = named.get(name);
            if (entry == null) {
                throw new IllegalArgumentException("Missing input: " + name);
            }
            return entry;
        }

        @Override
        public int size() {
            return ordered.size();
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T param(String name, Class<T> type) {
            KernelInputEntry entry = named.get(name);
            if (entry == null) {
                throw new IllegalArgumentException("Missing param: " + name);
            }
            if (entry.kind() != KernelInputKind.SCALAR) {
                throw new IllegalArgumentException("Param " + name + " is not a scalar");
            }
            if (!type.isInstance(entry.value())) {
                throw new IllegalArgumentException("Param " + name + " is not of type " + type);
            }
            return (T) entry.value();
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T scalar(int index, Class<T> type) {
            KernelInputEntry entry = ordered.get(index);
            if (entry.kind() != KernelInputKind.SCALAR) {
                throw new IllegalArgumentException("Input at index " + index + " is not a scalar");
            }
            if (!type.isInstance(entry.value())) {
                throw new IllegalArgumentException("Scalar at index " + index + " is not " + type);
            }
            return (T) entry.value();
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T scalar(String name, Class<T> type) {
            KernelInputEntry entry = named.get(name);
            if (entry == null) {
                throw new IllegalArgumentException("Missing scalar: " + name);
            }
            if (entry.kind() != KernelInputKind.SCALAR) {
                throw new IllegalArgumentException("Input " + name + " is not a scalar");
            }
            if (!type.isInstance(entry.value())) {
                throw new IllegalArgumentException("Scalar " + name + " is not " + type);
            }
            return (T) entry.value();
        }
    }

    private static DataType inferScalarType(Object value) {
        if (value instanceof Boolean) {
            return DataType.BOOL;
        }
        if (value instanceof Byte) {
            return DataType.I8;
        }
        if (value instanceof Short) {
            return DataType.I16;
        }
        if (value instanceof Integer) {
            return DataType.I32;
        }
        if (value instanceof Long) {
            return DataType.I64;
        }
        if (value instanceof Float) {
            return DataType.FP32;
        }
        if (value instanceof Double) {
            return DataType.FP64;
        }
        return null;
    }
}

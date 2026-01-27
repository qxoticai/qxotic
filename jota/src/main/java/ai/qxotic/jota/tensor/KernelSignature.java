package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import java.util.List;
import java.util.Set;

public record KernelSignature(
        String name,
        List<TensorDescriptor> inputs,
        List<TensorDescriptor> outputs,
        Set<DataType> supportedDtypes,
        Set<Device> supportedDevices) {

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public static final class Builder {
        private final String name;
        private final List<TensorDescriptor> inputs = new java.util.ArrayList<>();
        private final List<TensorDescriptor> outputs = new java.util.ArrayList<>();
        private final Set<DataType> dtypes = new java.util.HashSet<>();
        private final Set<Device> devices = new java.util.HashSet<>();

        Builder(String name) {
            this.name = name;
        }

        public Builder input(String name, int rank, DataType dtype, Layout layout) {
            inputs.add(new TensorDescriptor(name, rank, dtype, layout));
            return this;
        }

        public Builder output(String name, int rank, DataType dtype, Layout layout) {
            outputs.add(new TensorDescriptor(name, rank, dtype, layout));
            return this;
        }

        public Builder supportedDtypes(DataType... types) {
            dtypes.addAll(java.util.Arrays.asList(types));
            return this;
        }

        public Builder supportedDevices(Device... deviceTypes) {
            devices.addAll(java.util.Arrays.asList(deviceTypes));
            return this;
        }

        public KernelSignature build() {
            return new KernelSignature(name, List.copyOf(inputs), List.copyOf(outputs), Set.copyOf(dtypes), Set.copyOf(devices));
        }
    }
}

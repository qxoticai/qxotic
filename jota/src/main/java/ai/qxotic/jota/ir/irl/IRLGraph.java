package ai.qxotic.jota.ir.irl;

import java.util.List;
import java.util.Objects;

/**
 * Container for an IR-L program. An IR-L graph consists of input buffers, output buffers, and a
 * body that performs the computation.
 */
public record IRLGraph(List<BufferRef> inputs, List<BufferRef> outputs, IRLNode body) {

    public IRLGraph {
        Objects.requireNonNull(inputs, "inputs cannot be null");
        Objects.requireNonNull(outputs, "outputs cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
        inputs = List.copyOf(inputs);
        outputs = List.copyOf(outputs);
    }

    /** Returns the total number of buffers (inputs + outputs). */
    public int bufferCount() {
        return inputs.size() + outputs.size();
    }

    /** Returns the input buffer with the given index. */
    public BufferRef getInput(int index) {
        return inputs.get(index);
    }

    /** Returns the output buffer with the given index. */
    public BufferRef getOutput(int index) {
        return outputs.get(index);
    }

    /** Builder for constructing IR-L graphs. */
    public static class Builder {
        private final java.util.ArrayList<BufferRef> inputs = new java.util.ArrayList<>();
        private final java.util.ArrayList<BufferRef> outputs = new java.util.ArrayList<>();
        private int nextBufferId = 0;

        /** Adds an input buffer and returns its reference. */
        public BufferRef addInput(ai.qxotic.jota.DataType dtype, long[] shape, long[] strides) {
            BufferRef ref = new BufferRef(nextBufferId++, dtype, shape, strides);
            inputs.add(ref);
            return ref;
        }

        /** Adds a contiguous input buffer and returns its reference. */
        public BufferRef addContiguousInput(ai.qxotic.jota.DataType dtype, long... shape) {
            BufferRef ref = BufferRef.contiguous(nextBufferId++, dtype, shape);
            inputs.add(ref);
            return ref;
        }

        /** Adds an output buffer and returns its reference. */
        public BufferRef addOutput(ai.qxotic.jota.DataType dtype, long[] shape, long[] strides) {
            BufferRef ref = new BufferRef(nextBufferId++, dtype, shape, strides);
            outputs.add(ref);
            return ref;
        }

        /** Adds a contiguous output buffer and returns its reference. */
        public BufferRef addContiguousOutput(ai.qxotic.jota.DataType dtype, long... shape) {
            BufferRef ref = BufferRef.contiguous(nextBufferId++, dtype, shape);
            outputs.add(ref);
            return ref;
        }

        /** Builds the IR-L graph with the given body. */
        public IRLGraph build(IRLNode body) {
            return new IRLGraph(inputs, outputs, body);
        }
    }

    /** Creates a new builder. */
    public static Builder builder() {
        return new Builder();
    }
}

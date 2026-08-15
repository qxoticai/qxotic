package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Container for an IR-L program. An IR-L graph consists of inputs (buffers or scalars), output
 * buffers, and a body that performs the computation.
 */
public record LIRGraph(
        LIRExprGraph exprGraph, List<LIRInput> inputs, List<BufferRef> outputs, LIRExprNode body) {

    public LIRGraph {
        Objects.requireNonNull(exprGraph, "exprGraph cannot be null");
        Objects.requireNonNull(inputs, "inputs cannot be null");
        Objects.requireNonNull(outputs, "outputs cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
        inputs = List.copyOf(inputs);
        outputs = List.copyOf(outputs);
    }

    /** Returns the total number of inputs. */
    public int inputCount() {
        return inputs.size();
    }

    /** Returns the total number of outputs. */
    public int outputCount() {
        return outputs.size();
    }

    /** Returns the input with the given index. */
    public LIRInput getInput(int index) {
        return inputs.get(index);
    }

    /** Returns the output buffer with the given index. */
    public BufferRef getOutput(int index) {
        return outputs.get(index);
    }

    /** Builder for constructing IR-L graphs. */
    public static class Builder {
        private final LIRExprGraph exprGraph = new LIRExprGraph();
        private final List<LIRInput> inputs = new ArrayList<>();
        private final List<BufferRef> outputs = new ArrayList<>();
        private int nextId = 0;

        /** Adds an input buffer with the given layout and returns its reference. */
        public BufferRef addInput(DataType dtype, Layout layout) {
            BufferRef ref = BufferRef.of(nextId++, dtype, layout);
            inputs.add(ref);
            return ref;
        }

        /** Adds an input buffer with the given shape and strides and returns its reference. */
        public BufferRef addInput(DataType dtype, long[] shape, long[] strides) {
            Shape s = Shape.flat(shape);
            Stride st = Stride.flat(strides);
            BufferRef ref = BufferRef.of(nextId++, dtype, Layout.of(s, st));
            inputs.add(ref);
            return ref;
        }

        /** Adds a contiguous input buffer and returns its reference. */
        public BufferRef addContiguousInput(DataType dtype, long... shape) {
            BufferRef ref = BufferRef.contiguous(nextId++, dtype, shape);
            inputs.add(ref);
            return ref;
        }

        /** Adds a scalar input and returns its reference. */
        public ScalarInput addScalarInput(DataType dtype) {
            ScalarInput ref = new ScalarInput(nextId++, dtype);
            inputs.add(ref);
            return ref;
        }

        /** Adds an output buffer with the given layout and returns its reference. */
        public BufferRef addOutput(DataType dtype, Layout layout) {
            BufferRef ref = BufferRef.of(nextId++, dtype, layout);
            outputs.add(ref);
            return ref;
        }

        /** Adds an output buffer with the given shape and strides and returns its reference. */
        public BufferRef addOutput(DataType dtype, long[] shape, long[] strides) {
            Shape s = Shape.flat(shape);
            Stride st = Stride.flat(strides);
            BufferRef ref = BufferRef.of(nextId++, dtype, Layout.of(s, st));
            outputs.add(ref);
            return ref;
        }

        /** Adds a contiguous output buffer and returns its reference. */
        public BufferRef addContiguousOutput(DataType dtype, long... shape) {
            BufferRef ref = BufferRef.contiguous(nextId++, dtype, shape);
            outputs.add(ref);
            return ref;
        }

        public LIRExprGraph exprGraph() {
            return exprGraph;
        }

        /** Builds the IR-L graph with the given body. */
        public LIRGraph build(LIRExprNode body) {
            return new LIRGraph(exprGraph, inputs, outputs, body);
        }
    }

    /** Creates a new builder. */
    public static Builder builder() {
        return new Builder();
    }
}

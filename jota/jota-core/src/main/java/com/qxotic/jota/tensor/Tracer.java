package com.qxotic.jota.tensor;

import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.IotaConstant;
import com.qxotic.jota.ir.tir.RandomUniformOp;
import com.qxotic.jota.ir.tir.ScalarConstant;
import com.qxotic.jota.ir.tir.ScalarInput;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.random.RandomAlgorithms;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Orchestrates IR-T tracing and creates IRComputation. Similar to Tracer but uses IR-T instead of
 * ExprNode.
 */
public final class Tracer {

    private static final ScopedValue<TensorOps> IR_CONTEXT = ScopedValue.newInstance();

    private Tracer() {}

    /** Returns true if we are currently inside a {@code Tracer.trace()} call. */
    public static boolean isTracing() {
        return IR_CONTEXT.isBound();
    }

    /**
     * Returns the current {@link IRTensorOps} if inside a trace, for use by view ops and other
     * methods that need to call IRTensorOps directly inside a trace lambda.
     */
    static TensorOps requireIROps() {
        if (IR_CONTEXT.isBound()) {
            return IR_CONTEXT.get();
        }
        throw new IllegalStateException("Not inside a Tracer.trace() context");
    }

    public static <T> T withRequiredIROps(Function<TensorOps, T> fn) {
        Objects.requireNonNull(fn, "fn");
        return fn.apply(requireIROps());
    }

    /** Runs a supplier inside an IR context (package-private). */
    static <T> T withIR(Supplier<T> supplier) {
        Objects.requireNonNull(supplier, "supplier");
        try {
            return ScopedValue.where(IR_CONTEXT, new IRTensorOps()).call(supplier::get);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Scoped IR action failed", e);
        }
    }

    /** Traces a single-input function to IR-T. */
    public static Tensor trace(Tensor input, Function<Tensor, Tensor> fn) {
        return trace(List.of(input), tensors -> fn.apply(tensors.get(0)));
    }

    /** Traces a two-input function to IR-T. */
    public static Tensor trace(Tensor left, Tensor right, BiFunction<Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(fn, "fn");
        return trace(List.of(left, right), tensors -> fn.apply(tensors.get(0), tensors.get(1)));
    }

    public static Tensor trace(Supplier<Tensor> fn) {
        return trace(List.of(), unused -> fn.get());
    }

    /** Traces a multi-input function to IR-T. */
    public static Tensor trace(List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        TraceInputs traceInputs = traceInputs(inputs);
        Tensor output = withIR(() -> fn.apply(new ArrayList<>(traceInputs.tensors())));
        TIRGraph graph =
                new TIRGraph(traceInputs.nodes(), List.of(InternalTensorAccess.irNode(output)));
        return TensorFactory.lazy(
                new IRComputation(graph, inputs),
                output.dataType(),
                output.layout(),
                output.device());
    }

    private static TraceInputs traceInputs(List<Tensor> inputs) {
        List<TIRNode> nodes = new ArrayList<>();
        List<Tensor> tensors = new ArrayList<>();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            TIRNode node;
            boolean scalarBroadcast = input.layout().stride().isAllZeros();

            if (scalarBroadcast
                    && Environment.runtimeFor(input.device())
                            .supportsNativeRuntimeAlias()) {
                node = new ScalarInput(i, input.dataType(), input.shape());
            } else if (InternalTensorAccess.isMaterialized(input)
                    || InternalTensorAccess.computation(input).isEmpty()) {
                // Materialized tensor - create a TensorInput
                node = new TensorInput(i, input.dataType(), input.layout());
            } else {
                LazyComputation comp = InternalTensorAccess.computation(input).orElseThrow();
                if (comp instanceof RangeComputation range) {
                    // Iota constant - computed from loop index
                    node =
                            new IotaConstant(
                                    range.count(), input.dataType(), Shape.flat(range.count()));
                } else if (comp instanceof RandomComputation random) {
                    node =
                            new RandomUniformOp(
                                    random.shape(),
                                    random.dataType(),
                                    RandomAlgorithms.key0(random.key()),
                                    RandomAlgorithms.key1(random.key()));
                } else if (comp instanceof ConstantComputation constComp) {
                    // Constant value - inline as ScalarConstant
                    node =
                            ScalarConstant.broadcast(
                                    constComp.rawBits(),
                                    constComp.dataType(),
                                    input.layout().shape());
                } else {
                    // IRComputation or other - treat as buffer input
                    // The computation will be executed first, and its result used as input
                    node = new TensorInput(i, input.dataType(), input.layout());
                }
            }

            nodes.add(node);
            tensors.add(new IRTensorImpl(node, input.device()));
        }

        return new TraceInputs(nodes, tensors);
    }

    private record TraceInputs(List<TIRNode> nodes, List<Tensor> tensors) {
        public List<TIRNode> nodes() {
            return nodes;
        }

        public List<Tensor> tensors() {
            return tensors;
        }
    }
}

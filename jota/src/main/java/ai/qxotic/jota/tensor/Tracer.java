package ai.qxotic.jota.tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

public final class Tracer {

    private Tracer() {}

    public static OptimizingCallSite createOptimizingCallSite(Function<Tensor, Tensor> function) {
        Objects.requireNonNull(function, "function");
        return new OptimizingCallSiteImpl(function);
    }

    public static Tensor trace(Tensor input, Function<Tensor, Tensor> function) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(function, "function");

        TraceInputs inputs = traceInputs(List.of(input));
        TraceTensor tracedInput = inputs.traces().get(0);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output =
                TensorOpsContext.with(
                        tracingOps,
                        () -> {
                            Tensor result = function.apply(tracedInput);
                            if (result instanceof TraceTensor traceResult) {
                                return traceResult;
                            }
                            throw new IllegalStateException(
                                    "Tracing function must return a traced tensor, got: " + result);
                        });

        ExpressionGraph graph = new ExpressionGraph(output.node(), inputs.nodes());
        ExpressionComputation computation = new ExpressionComputation(graph, inputs.tensors());
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }

    public static Tensor trace(
            Tensor left, Tensor right, java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(fn, "fn");

        TraceInputs inputs = traceInputs(List.of(left, right));
        TraceTensor tracedLeft = inputs.traces().get(0);
        TraceTensor tracedRight = inputs.traces().get(1);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output =
                TensorOpsContext.with(
                        tracingOps,
                        () -> {
                            Tensor result = fn.apply(tracedLeft, tracedRight);
                            if (result instanceof TraceTensor traceResult) {
                                return traceResult;
                            }
                            throw new IllegalStateException(
                                    "Tracing function must return a traced tensor, got: " + result);
                        });

        ExpressionGraph graph = new ExpressionGraph(output.node(), inputs.nodes());
        ExpressionComputation computation = new ExpressionComputation(graph, inputs.tensors());
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }

    public static Tensor trace(
            Tensor first,
            Tensor second,
            Tensor third,
            TriFunction<Tensor, Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(third, "third");
        Objects.requireNonNull(fn, "fn");

        TraceInputs inputs = traceInputs(List.of(first, second, third));
        TraceTensor tracedFirst = inputs.traces().get(0);
        TraceTensor tracedSecond = inputs.traces().get(1);
        TraceTensor tracedThird = inputs.traces().get(2);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output =
                TensorOpsContext.with(
                        tracingOps,
                        () -> {
                            Tensor result = fn.apply(tracedFirst, tracedSecond, tracedThird);
                            if (result instanceof TraceTensor traceResult) {
                                return traceResult;
                            }
                            throw new IllegalStateException(
                                    "Tracing function must return a traced tensor, got: " + result);
                        });

        ExpressionGraph graph = new ExpressionGraph(output.node(), inputs.nodes());
        ExpressionComputation computation = new ExpressionComputation(graph, inputs.tensors());
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }

    private static TraceInputs traceInputs(List<Tensor> inputs) {
        List<InputNode> nodes = new ArrayList<>();
        List<Tensor> tensors = new ArrayList<>();
        List<TraceTensor> traces = new ArrayList<>(inputs.size());
        int inputIndex = 0;
        for (Tensor input : inputs) {
            if (input.computation().orElse(null) instanceof RangeComputation range) {
                TraceTensor trace =
                        new TraceTensor(
                                new RangeNode(range.count(), input.layout(), input.device()));
                traces.add(trace);
                continue;
            }
            InputNode node =
                    new InputNode(inputIndex, input.dataType(), input.layout(), input.device());
            nodes.add(node);
            tensors.add(input);
            traces.add(new TraceTensor(node));
            inputIndex++;
        }
        return new TraceInputs(traces, nodes, tensors);
    }

    private record TraceInputs(
            List<TraceTensor> traces, List<InputNode> nodes, List<Tensor> tensors) {}
}

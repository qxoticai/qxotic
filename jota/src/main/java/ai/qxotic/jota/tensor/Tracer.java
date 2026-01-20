package ai.qxotic.jota.tensor;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

public final class Tracer {

    private Tracer() {}

    public static Tensor trace(Tensor input, Function<Tensor, Tensor> function) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(function, "function");

        InputNode inputNode =
                new InputNode(0, input.dataType(), input.layout(), input.device());
        TraceTensor tracedInput = new TraceTensor(inputNode);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output = TensorOpsContext.with(
                tracingOps,
                () -> {
                    Tensor result = function.apply(tracedInput);
                    if (result instanceof TraceTensor traceResult) {
                        return traceResult;
                    }
                    throw new IllegalStateException(
                            "Tracing function must return a traced tensor, got: " + result);
                });

        ExpressionGraph graph = new ExpressionGraph(output.node(), List.of(inputNode));
        ExpressionComputation computation = new ExpressionComputation(graph, List.of(input));
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }

    public static Tensor trace(
            Tensor left, Tensor right, java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(fn, "fn");

        InputNode leftNode = new InputNode(0, left.dataType(), left.layout(), left.device());
        InputNode rightNode = new InputNode(1, right.dataType(), right.layout(), right.device());
        TraceTensor tracedLeft = new TraceTensor(leftNode);
        TraceTensor tracedRight = new TraceTensor(rightNode);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output = TensorOpsContext.with(
                tracingOps,
                () -> {
                    Tensor result = fn.apply(tracedLeft, tracedRight);
                    if (result instanceof TraceTensor traceResult) {
                        return traceResult;
                    }
                    throw new IllegalStateException(
                            "Tracing function must return a traced tensor, got: " + result);
                });

        ExpressionGraph graph = new ExpressionGraph(output.node(), List.of(leftNode, rightNode));
        ExpressionComputation computation =
                new ExpressionComputation(graph, List.of(left, right));
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }

    public static Tensor trace(
            Tensor first, Tensor second, Tensor third, TriFunction<Tensor, Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(third, "third");
        Objects.requireNonNull(fn, "fn");

        InputNode firstNode = new InputNode(0, first.dataType(), first.layout(), first.device());
        InputNode secondNode = new InputNode(1, second.dataType(), second.layout(), second.device());
        InputNode thirdNode = new InputNode(2, third.dataType(), third.layout(), third.device());
        TraceTensor tracedFirst = new TraceTensor(firstNode);
        TraceTensor tracedSecond = new TraceTensor(secondNode);
        TraceTensor tracedThird = new TraceTensor(thirdNode);
        TracingTensorOps tracingOps = new TracingTensorOps();

        TraceTensor output = TensorOpsContext.with(
                tracingOps,
                () -> {
                    Tensor result = fn.apply(tracedFirst, tracedSecond, tracedThird);
                    if (result instanceof TraceTensor traceResult) {
                        return traceResult;
                    }
                    throw new IllegalStateException(
                            "Tracing function must return a traced tensor, got: " + result);
                });

        ExpressionGraph graph =
                new ExpressionGraph(output.node(), List.of(firstNode, secondNode, thirdNode));
        ExpressionComputation computation =
                new ExpressionComputation(graph, List.of(first, second, third));
        return Tensor.lazy(computation, output.dataType(), output.layout(), output.device());
    }
}

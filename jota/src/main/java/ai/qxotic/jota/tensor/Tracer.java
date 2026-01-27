package ai.qxotic.jota.tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
            LazyComputation computation = input.computation().orElse(null);
            if (computation instanceof RangeComputation range) {
                TraceTensor trace =
                        new TraceTensor(
                                new RangeNode(range.count(), input.layout(), input.device()));
                traces.add(trace);
                continue;
            }
            if (computation instanceof ConstantComputation constant) {
                ScalarNode node =
                        new ScalarNode(
                                constant.value(), input.dataType(), input.layout(), input.device());
                traces.add(new TraceTensor(node));
                continue;
            }
            if (computation instanceof ExpressionComputation expr) {
                ExpressionGraph graph = expr.graph();
                if (containsBoundaryNode(graph.root())) {
                    InputNode node =
                            new InputNode(
                                    inputIndex, input.dataType(), input.layout(), input.device());
                    nodes.add(node);
                    tensors.add(input);
                    traces.add(new TraceTensor(node));
                    inputIndex++;
                    continue;
                }
                List<InputNode> subInputs = graph.inputs();
                Map<InputNode, InputNode> remap = new HashMap<>();
                for (InputNode subInput : subInputs) {
                    InputNode mapped =
                            new InputNode(
                                    inputIndex,
                                    subInput.dataType(),
                                    subInput.layout(),
                                    subInput.device());
                    remap.put(subInput, mapped);
                    nodes.add(mapped);
                    inputIndex++;
                }
                tensors.addAll(expr.inputs());
                ExprNode remappedRoot = remapInputs(graph.root(), remap, new HashMap<>());
                traces.add(new TraceTensor(remappedRoot));
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

    private static ExprNode remapInputs(
            ExprNode node, Map<InputNode, InputNode> remap, Map<ExprNode, ExprNode> cache) {
        ExprNode cached = cache.get(node);
        if (cached != null) {
            return cached;
        }
        ExprNode mapped;
        if (node instanceof InputNode input) {
            mapped = remap.getOrDefault(input, input);
        } else if (node instanceof ScalarNode
                || node instanceof RangeNode) {
            mapped = node;
        } else if (node instanceof UnaryNode unary) {
            ExprNode child = remapInputs(unary.input(), remap, cache);
            mapped = new UnaryNode(unary.op(), child, unary.dataType(), unary.layout(), unary.device());
        } else if (node instanceof BinaryNode binary) {
            ExprNode left = remapInputs(binary.left(), remap, cache);
            ExprNode right = remapInputs(binary.right(), remap, cache);
            mapped =
                    new BinaryNode(
                            binary.op(), left, right, binary.dataType(), binary.layout(), binary.device());
        } else if (node instanceof TernaryNode ternary) {
            ExprNode cond = remapInputs(ternary.condition(), remap, cache);
            ExprNode tVal = remapInputs(ternary.trueValue(), remap, cache);
            ExprNode fVal = remapInputs(ternary.falseValue(), remap, cache);
            mapped =
                    new TernaryNode(
                            ternary.op(),
                            cond,
                            tVal,
                            fVal,
                            ternary.dataType(),
                            ternary.layout(),
                            ternary.device());
        } else if (node instanceof TransferNode transfer) {
            ExprNode child = remapInputs(transfer.input(), remap, cache);
            mapped =
                    new TransferNode(child, transfer.targetDevice(), transfer.dataType(), transfer.layout());
        } else if (node instanceof ContiguousNode contiguous) {
            ExprNode child = remapInputs(contiguous.input(), remap, cache);
            mapped = new ContiguousNode(child, contiguous.dataType(), contiguous.layout(), contiguous.device());
        } else if (node instanceof CastNode cast) {
            ExprNode child = remapInputs(cast.input(), remap, cache);
            mapped = new CastNode(child, cast.targetType(), cast.layout(), cast.device());
        } else if (node instanceof ViewTransformOp transform) {
            ExprNode child = remapInputs(transform.input(), remap, cache);
            mapped =
                    new ViewTransformOp(
                            child,
                            transform.layout(),
                            transform.byteOffsetDelta(),
                            transform.hint(),
                            transform.dataType(),
                            transform.device());
        } else if (node instanceof ReductionNode reduction) {
            ExprNode child = remapInputs(reduction.input(), remap, cache);
            mapped =
                    new ReductionNode(
                            reduction.op(),
                            child,
                            reduction.axis(),
                            reduction.keepDims(),
                            reduction.dataType(),
                            reduction.layout(),
                            reduction.device());
        } else {
            throw new UnsupportedOperationException("Unsupported node remap: " + node.getClass());
        }
        cache.put(node, mapped);
        return mapped;
    }

    private static boolean containsBoundaryNode(ExprNode node) {
        java.util.ArrayDeque<ExprNode> stack = new java.util.ArrayDeque<>();
        stack.push(node);
        while (!stack.isEmpty()) {
            ExprNode current = stack.pop();
            if (current instanceof TransferNode
                    || current instanceof ContiguousNode
                    || current instanceof ViewTransformOp) {
                return true;
            }
            if (current instanceof UnaryNode unary) {
                stack.push(unary.input());
            } else if (current instanceof BinaryNode binary) {
                stack.push(binary.left());
                stack.push(binary.right());
            } else if (current instanceof TernaryNode ternary) {
                stack.push(ternary.condition());
                stack.push(ternary.trueValue());
                stack.push(ternary.falseValue());
            } else if (current instanceof CastNode cast) {
                stack.push(cast.input());
            } else if (current instanceof ReductionNode reduction) {
                stack.push(reduction.input());
            } else if (current instanceof InputNode
                    || current instanceof ScalarNode
                    || current instanceof RangeNode) {
                continue;
            } else {
                throw new IllegalStateException("Unsupported node: " + current);
            }
        }
        return false;
    }

    private record TraceInputs(
            List<TraceTensor> traces, List<InputNode> nodes, List<Tensor> tensors) {}
}

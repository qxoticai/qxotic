package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class ExpressionComputation implements LazyComputation {

    private final ExpressionGraph graph;
    private final List<Tensor> inputs;

    ExpressionComputation(ExpressionGraph graph, List<Tensor> inputs) {
        this.graph = Objects.requireNonNull(graph, "graph");
        this.inputs = List.copyOf(inputs);
    }

    ExpressionGraph graph() {
        return graph;
    }

    @Override
    public Op operation() {
        return ExpressionOp.INSTANCE;
    }

    @Override
    public List<Tensor> inputs() {
        return inputs;
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("graph", graph);
    }

    @Override
    public MemoryView<?> execute() {
        ExprNode root = graph.root();
        if (containsBoundaryNode(root)) {
            if (root instanceof TransferNode transfer) {
                return executeTransfer(transfer);
            }
            if (root instanceof ContiguousNode contiguous) {
                return executeContiguous(contiguous);
            }
            if (root instanceof ViewTransformOp transform) {
                return executeViewTransform(transform);
            }
            throw new IllegalStateException(
                    "View transform/transfer/contiguous operations must be the final node in the graph");
        }
        ComputeEngine engine = resolveEngine(root.device());
        ComputeBackend backend = engine.backendFor(root.device());
        return backend.execute(graph, inputs);
    }

    private boolean containsBoundaryNode(ExprNode node) {
        Deque<ExprNode> stack = new ArrayDeque<>();
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
            } else if (current instanceof ViewTransformOp transform) {
                stack.push(transform.input());
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

    private MemoryView<?> executeTransfer(TransferNode transfer) {
        ExprNode inputNode = transfer.input();
        ExpressionGraph sourceGraph = new ExpressionGraph(inputNode, graph.inputs());
        ComputeEngine engine = resolveEngine(inputNode.device());
        MemoryView<?> sourceView =
                engine.backendFor(inputNode.device()).execute(sourceGraph, inputs);
        if (inputNode.device().equals(transfer.device())) {
            return sourceView;
        }
        if (sourceView.dataType() != transfer.dataType()) {
            throw new IllegalStateException(
                    "Transfer dtype mismatch: "
                            + sourceView.dataType()
                            + " vs "
                            + transfer.dataType());
        }
        if (!transfer.layout().isCongruentWith(sourceView.layout())) {
            throw new IllegalStateException(
                    "Transfer layout mismatch: "
                            + transfer.layout()
                            + " vs "
                            + sourceView.layout());
        }
        MemoryContext<?> sourceContext =
                Environment.current().registry().context(inputNode.device());
        MemoryContext<?> targetContext =
                Environment.current().registry().context(transfer.device());
        if (!targetContext.supportsDataType(sourceView.dataType())) {
            throw new IllegalArgumentException(
                    "Target context does not support data type: " + sourceView.dataType());
        }
        if (!sourceContext.device().equals(targetContext.device()) && !sourceView.isContiguous()) {
            throw new IllegalArgumentException(
                    "Target backend cannot preserve layout; call x.contiguous().to(device)");
        }
        OutputBufferSpec outputSpec = computeOutputSpec(transfer.layout(), transfer.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        targetContext.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        transfer.dataType(),
                        transfer.layout());
        copyBetweenContexts(sourceContext, sourceView, targetContext, targetView);
        return targetView;
    }

    private MemoryView<?> executeContiguous(ContiguousNode contiguous) {
        ExprNode inputNode = contiguous.input();
        ExpressionGraph sourceGraph = new ExpressionGraph(inputNode, graph.inputs());
        ComputeEngine engine = resolveEngine(inputNode.device());
        MemoryView<?> sourceView =
                engine.backendFor(inputNode.device()).execute(sourceGraph, inputs);
        if (!contiguous.layout().isCongruentWith(sourceView.layout())) {
            throw new IllegalStateException(
                    "Contiguous layout mismatch: "
                            + contiguous.layout()
                            + " vs "
                            + sourceView.layout());
        }
        if (sourceView.isContiguous()) {
            return sourceView;
        }
        MemoryContext<?> context = Environment.current().registry().context(inputNode.device());
        if (!context.supportsDataType(sourceView.dataType())) {
            throw new IllegalArgumentException(
                    "Target context does not support data type: " + sourceView.dataType());
        }
        OutputBufferSpec outputSpec = computeOutputSpec(contiguous.layout(), contiguous.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        contiguous.dataType(),
                        contiguous.layout());
        copyBetweenContexts(context, sourceView, context, targetView);
        return targetView;
    }

    private MemoryView<?> executeViewTransform(ViewTransformOp transform) {
        ExprNode inputNode = transform.input();
        ExpressionGraph sourceGraph = new ExpressionGraph(inputNode, graph.inputs());
        ComputeEngine engine = resolveEngine(inputNode.device());
        MemoryView<?> sourceView =
                engine.backendFor(inputNode.device()).execute(sourceGraph, inputs);
        if (sourceView.dataType() != transform.dataType()) {
            throw new IllegalStateException(
                    "View transform dtype mismatch: "
                            + sourceView.dataType()
                            + " vs "
                            + transform.dataType());
        }
        if (!inputNode.device().equals(transform.device())) {
            throw new IllegalStateException(
                    "View transform device mismatch: "
                            + inputNode.device()
                            + " vs "
                            + transform.device());
        }
        return MemoryView.of(
                sourceView.memory(),
                sourceView.byteOffset() + transform.byteOffsetDelta(),
                sourceView.dataType(),
                transform.layout());
    }

    private static void copyBetweenContexts(
            MemoryContext<?> sourceContext,
            MemoryView<?> sourceView,
            MemoryContext<?> targetContext,
            MemoryView<?> targetView) {
        @SuppressWarnings("unchecked")
        MemoryContext<Object> source = (MemoryContext<Object>) sourceContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> src = (MemoryView<Object>) sourceView;
        @SuppressWarnings("unchecked")
        MemoryContext<Object> target = (MemoryContext<Object>) targetContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> dst = (MemoryView<Object>) targetView;
        MemoryContext.copy(source, src, target, dst);
    }

    private static OutputBufferSpec computeOutputSpec(Layout layout, DataType dataType) {
        long[] shape = layout.shape().toArray();
        long[] strideBytes = layout.stride().scale(dataType.byteSize()).toArray();
        long minOffset = 0;
        long maxOffset = 0;
        for (int i = 0; i < shape.length; i++) {
            long dim = shape[i];
            if (dim <= 1) {
                continue;
            }
            long span = (dim - 1) * strideBytes[i];
            if (strideBytes[i] >= 0) {
                maxOffset += span;
            } else {
                minOffset += span;
            }
        }
        long byteOffset = -minOffset;
        long byteSize = maxOffset - minOffset + dataType.byteSize();
        return new OutputBufferSpec(byteOffset, byteSize);
    }

    private static ComputeEngine resolveEngine(Device device) {
        ComputeEngine engine = ComputeEngineContext.current();
        if (engine != null) {
            return engine;
        }
        return Environment.current().engineFor(device);
    }

    private record OutputBufferSpec(long byteOffset, long byteSize) {}
}

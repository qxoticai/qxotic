package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class ExpressionComputation implements LazyComputation {

    private final ExpressionGraph graph;
    private final List<Tensor> inputs;
    private final java.util.Map<Integer, Tensor> inputTensorMap;

    ExpressionComputation(
            ExpressionGraph graph,
            List<Tensor> inputs,
            java.util.Map<Integer, Tensor> inputTensorMap) {
        this.graph = Objects.requireNonNull(graph, "graph");
        this.inputs = List.copyOf(inputs);
        this.inputTensorMap = java.util.Map.copyOf(inputTensorMap);
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

    java.util.Map<Integer, Tensor> inputTensorMap() {
        return inputTensorMap;
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("graph", graph);
    }

    @Override
    public MemoryView<?> execute() {
        return executeNode(graph.root());
    }

    private MemoryView<?> executeTransfer(TransferNode transfer) {
        ExprNode inputNode = transfer.input();
        MemoryView<?> sourceView = executeNode(inputNode);
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
                Environment.current().backend(inputNode.device()).memoryContext();
        MemoryContext<?> targetContext =
                Environment.current().backend(transfer.device()).memoryContext();
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
        MemoryView<?> sourceView = executeNode(inputNode);
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
        MemoryContext<?> context =
                Environment.current().backend(inputNode.device()).memoryContext();
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
        MemoryView<?> sourceView = executeNode(inputNode);
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

    private MemoryView<?> executeNode(ExprNode node) {
        if (node instanceof TransferNode transfer) {
            return executeTransfer(transfer);
        }
        if (node instanceof ContiguousNode contiguous) {
            return executeContiguous(contiguous);
        }
        if (node instanceof ViewTransformOp transform) {
            return executeViewTransform(transform);
        }
        ExpressionGraph subGraph =
                new ExpressionGraph(node, graph.inputs(), graph.inputTensorMap());
        ComputeEngine engine = resolveEngine(node.device());
        ComputeBackend backend = engine.backendFor(node.device());
        return backend.execute(subGraph, inputs);
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
        return Environment.current().engineFor(device);
    }

    private record OutputBufferSpec(long byteOffset, long byteSize) {}
}

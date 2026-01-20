package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class JavaComputeBackend implements ComputeBackend {

    private final MemoryContext<?> context;
    private final KernelCache cache;
    private final JavaKernelCompiler compiler;

    JavaComputeBackend(MemoryContext<?> context, KernelCache cache) {
        this.context = Objects.requireNonNull(context, "context");
        this.cache = Objects.requireNonNull(cache, "cache");
        this.compiler = new JavaKernelCompiler(cache);
    }


    @Override
    public Device device() {
        return context.device();
    }

    @Override
    public MemoryView<?> execute(ExpressionGraph graph, List<Tensor> inputs) {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(inputs, "inputs");
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        List<MemoryView<MemorySegment>> inputViews = new ArrayList<>(inputs.size());
        boolean allContiguous = true;
        for (InputNode inputNode : graph.inputs()) {
            Tensor inputTensor = inputs.get(inputNode.index());
            MemoryView<?> view = inputTensor.materialize();
            requireCompatible(inputNode, view);
            requireSameShape(inputNode, view);
            requirePanama(view, "input" + inputNode.index());
            allContiguous &= view.isContiguous();
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> panamaView = (MemoryView<MemorySegment>) view;
            inputViews.add(panamaView);
        }

        ExprNode root = graph.root();
        Layout layout = root.layout();
        DataType dataType = root.dataType();
        OutputBufferSpec outputSpec = computeOutputSpec(layout, dataType);
        MemoryView<?> outputView =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(outputSpec.byteSize()),
                        outputSpec.byteOffset(),
                        dataType,
                        layout);
        requirePanama(outputView, "output");
        allContiguous &= outputView.isContiguous();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> panamaOutput = (MemoryView<MemorySegment>) outputView;

        KernelStyle style = allContiguous ? KernelStyle.CONTIGUOUS : KernelStyle.STRIDED;
        JitKernel kernel = compiler.compile(graph, style);
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] inputsArray =
                inputViews.toArray(size -> (MemoryView<MemorySegment>[]) new MemoryView[size]);
        kernel.execute(context, inputsArray, panamaOutput);
        return outputView;
    }

    private void requireCompatible(InputNode node, MemoryView<?> view) {
        if (view.dataType() != node.dataType()) {
            throw new IllegalArgumentException(
                    "Input dtype mismatch for input "
                            + node.index()
                            + ": expected "
                            + node.dataType()
                            + " but got "
                            + view.dataType());
        }
        if (!node.layout().isCongruentWith(view.layout())) {
            throw new IllegalArgumentException(
                    "Input layout mismatch for input "
                            + node.index()
                            + ": expected "
                            + node.layout()
                            + " but got "
                            + view.layout());
        }
        if (!node.device().equals(view.memory().device())) {
            throw new IllegalArgumentException(
                    "Input device mismatch for input "
                            + node.index()
                            + ": expected "
                            + node.device()
                            + " but got "
                            + view.memory().device());
        }
    }

    private void requireSameShape(InputNode node, MemoryView<?> view) {
        long[] expected = node.layout().shape().toArray();
        long[] actual = view.layout().shape().toArray();
        if (!java.util.Arrays.equals(expected, actual)) {
            throw new IllegalArgumentException(
                    "Input shape mismatch for input "
                            + node.index()
                            + ": expected "
                            + node.layout().shape()
                            + " but got "
                            + view.layout().shape());
        }
    }

    private void requirePanama(MemoryView<?> view, String label) {
        if (!(view.memory().base() instanceof MemorySegment)) {
            throw new IllegalArgumentException(
                    "Java backend requires Panama MemorySegment for " + label);
        }
    }

    private OutputBufferSpec computeOutputSpec(Layout layout, DataType dataType) {
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

    private record OutputBufferSpec(long byteOffset, long byteSize) {}
}

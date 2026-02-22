package com.qxotic.jota.runtime;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Util;
import com.qxotic.jota.ir.tir.GatherOp;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.BinaryOp;
import com.qxotic.jota.tensor.ExecutionStream;
import com.qxotic.jota.tensor.KernelArgs;
import com.qxotic.jota.tensor.KernelExecutable;
import com.qxotic.jota.tensor.LaunchConfig;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.UnaryOp;

final class LoadOnlyEagerKernels implements DeviceRuntime.EagerKernels {

    private final DeviceRuntime runtime;

    LoadOnlyEagerKernels(DeviceRuntime runtime) {
        this.runtime = runtime;
    }

    @Override
    public Tensor unary(UnaryOp op, Tensor x) {
        Tensor input = ensureContiguous(x, "unary");
        DataType dtype = input.dataType();
        Tensor output = allocate(dtype, input.shape());
        String kernelName =
                "eager.unary." + op.name().toLowerCase() + "." + dtype.name().toLowerCase();
        launch(kernelName, output, input, input.shape().size());
        return output;
    }

    @Override
    public Tensor binary(BinaryOp op, Tensor a, Tensor b) {
        Tensor left = ensureContiguous(a, "binary");
        Tensor right = ensureContiguous(b, "binary");
        DataType dtype = left.dataType();
        if (dtype != right.dataType()) {
            throw new UnsupportedOperationException(
                    "Load-only eager binary kernels require matching input dtypes");
        }
        if (!left.shape().equals(right.shape())) {
            throw new UnsupportedOperationException(
                    "Load-only eager binary kernels require matching input shapes");
        }
        Tensor output = allocate(dtype, left.shape());
        String kernelName =
                "eager.binary." + op.name().toLowerCase() + "." + dtype.name().toLowerCase();
        launch(kernelName, output, left, right, left.shape().size());
        return output;
    }

    @Override
    public Tensor compare(BinaryOp op, Tensor a, Tensor b) {
        Tensor left = ensureContiguous(a, "compare");
        Tensor right = ensureContiguous(b, "compare");
        if (left.dataType() != right.dataType()) {
            throw new UnsupportedOperationException(
                    "Load-only eager compare kernels require matching input dtypes");
        }
        if (!left.shape().equals(right.shape())) {
            throw new UnsupportedOperationException(
                    "Load-only eager compare kernels require matching input shapes");
        }
        Tensor output = allocate(DataType.BOOL, left.shape());
        String kernelName =
                "eager.compare."
                        + op.name().toLowerCase()
                        + "."
                        + left.dataType().name().toLowerCase();
        launch(kernelName, output, left, right, left.shape().size());
        return output;
    }

    @Override
    public Tensor logical(BinaryOp op, Tensor a, Tensor b) {
        Tensor left = ensureContiguous(a, "logical");
        Tensor right = ensureContiguous(b, "logical");
        if (left.dataType() != DataType.BOOL || right.dataType() != DataType.BOOL) {
            throw new UnsupportedOperationException(
                    "Load-only eager logical kernels require BOOL inputs");
        }
        if (!left.shape().equals(right.shape())) {
            throw new UnsupportedOperationException(
                    "Load-only eager logical kernels require matching input shapes");
        }
        Tensor output = allocate(DataType.BOOL, left.shape());
        String kernelName = "eager.logical." + op.name().toLowerCase();
        launch(kernelName, output, left, right, left.shape().size());
        return output;
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        Tensor input = ensureContiguous(x, "cast");
        Tensor output = allocate(targetType, input.shape());
        String kernelName =
                "eager.cast."
                        + input.dataType().name().toLowerCase()
                        + "."
                        + targetType.name().toLowerCase();
        launch(kernelName, output, input, input.shape().size());
        return output;
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        Tensor cond = ensureContiguous(condition, "where");
        Tensor whenTrue = ensureContiguous(trueValue, "where");
        Tensor whenFalse = ensureContiguous(falseValue, "where");
        if (cond.dataType() != DataType.BOOL) {
            throw new UnsupportedOperationException(
                    "Load-only eager where kernels require BOOL condition");
        }
        if (whenTrue.dataType() != whenFalse.dataType()) {
            throw new UnsupportedOperationException(
                    "Load-only eager where kernels require matching value dtypes");
        }
        if (!cond.shape().equals(whenTrue.shape()) || !whenTrue.shape().equals(whenFalse.shape())) {
            throw new UnsupportedOperationException(
                    "Load-only eager where kernels require matching input shapes");
        }
        Tensor output = allocate(whenTrue.dataType(), whenTrue.shape());
        String kernelName = "eager.where." + whenTrue.dataType().name().toLowerCase();
        launch(kernelName, output, cond, whenTrue, whenFalse, whenTrue.shape().size());
        return output;
    }

    @Override
    public Tensor reduce(
            DeviceRuntime.ReductionOp op,
            Tensor x,
            DataType accumulatorType,
            boolean keepDims,
            int axis,
            int... axes) {
        Tensor input = ensureContiguous(x, "reduce");
        int[] normalizedAxes = normalizeAxes(input.shape().rank(), axis, axes);
        Shape outputShape = reduceShape(input.shape(), normalizedAxes, keepDims);
        DataType outputType =
                switch (op) {
                    case SUM, PRODUCT ->
                            accumulatorType == null ? input.dataType() : accumulatorType;
                    case MIN, MAX -> input.dataType();
                };
        Tensor output = allocate(outputType, outputShape);
        String kernelName =
                "eager.reduce."
                        + op.name().toLowerCase()
                        + "."
                        + input.dataType().name().toLowerCase()
                        + "."
                        + outputType.name().toLowerCase();

        Object[] argsForKernel = new Object[8 + normalizedAxes.length + input.shape().rank()];
        int idx = 0;
        argsForKernel[idx++] = output;
        argsForKernel[idx++] = input;
        argsForKernel[idx++] = keepDims;
        argsForKernel[idx++] = input.shape().rank();
        argsForKernel[idx++] = output.shape().rank();
        argsForKernel[idx++] = normalizedAxes.length;
        argsForKernel[idx++] = input.shape().size();
        argsForKernel[idx++] = output.shape().size();
        for (int normalizedAxis : normalizedAxes) {
            argsForKernel[idx++] = normalizedAxis;
        }
        for (int i = 0; i < input.shape().rank(); i++) {
            argsForKernel[idx++] = input.shape().flatAt(i);
        }
        launch(kernelName, argsForKernel);
        return output;
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        Tensor left = ensureContiguous(a, "matmul");
        Tensor right = ensureContiguous(b, "matmul");
        if (left.dataType() != right.dataType()) {
            throw new UnsupportedOperationException(
                    "Load-only eager matmul kernels require matching input dtypes");
        }
        if (left.shape().rank() != 2 || right.shape().rank() != 2) {
            throw new UnsupportedOperationException(
                    "Load-only eager matmul kernels require rank-2 inputs");
        }
        long m = left.shape().flatAt(0);
        long k = left.shape().flatAt(1);
        if (k != right.shape().flatAt(0)) {
            throw new UnsupportedOperationException(
                    "Load-only eager matmul inner dimensions mismatch");
        }
        long n = right.shape().flatAt(1);
        Tensor output = allocate(left.dataType(), Shape.of(m, n));
        String kernelName = "eager.matmul." + left.dataType().name().toLowerCase();
        launch(kernelName, output, left, right, m, n, k);
        return output;
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        Tensor left = ensureContiguous(a, "batchedMatmul");
        Tensor right = ensureContiguous(b, "batchedMatmul");
        if (left.dataType() != right.dataType()) {
            throw new UnsupportedOperationException(
                    "Load-only eager batchedMatmul kernels require matching input dtypes");
        }
        if (left.shape().rank() != 3 || right.shape().rank() != 3) {
            throw new UnsupportedOperationException(
                    "Load-only eager batchedMatmul kernels require rank-3 inputs");
        }
        long batch = left.shape().flatAt(0);
        long m = left.shape().flatAt(1);
        long k = left.shape().flatAt(2);
        if (batch != right.shape().flatAt(0) || k != right.shape().flatAt(1)) {
            throw new UnsupportedOperationException(
                    "Load-only eager batchedMatmul dimensions mismatch");
        }
        long n = right.shape().flatAt(2);
        Tensor output = allocate(left.dataType(), Shape.of(batch, m, n));
        String kernelName = "eager.batched_matmul." + left.dataType().name().toLowerCase();
        launch(kernelName, output, left, right, batch, m, n, k);
        return output;
    }

    @Override
    public Tensor gather(Tensor input, Tensor indices, int axis) {
        Tensor source = ensureContiguous(input, "gather");
        Tensor indexTensor = ensureContiguous(indices, "gather");
        if (!indexTensor.dataType().isIntegral()) {
            throw new UnsupportedOperationException(
                    "Load-only eager gather kernels require integral indices");
        }
        int normalizedAxis = Util.wrapAround(axis, source.shape().rank());
        Shape outputShape =
                GatherOp.computeOutputShape(source.shape(), indexTensor.shape(), normalizedAxis);
        Tensor output = allocate(source.dataType(), outputShape);
        String kernelName =
                "eager.gather."
                        + source.dataType().name().toLowerCase()
                        + "."
                        + indexTensor.dataType().name().toLowerCase();

        Object[] argsForKernel =
                new Object
                        [10
                                + source.shape().rank()
                                + indexTensor.shape().rank()
                                + output.shape().rank()];
        int idx = 0;
        argsForKernel[idx++] = output;
        argsForKernel[idx++] = source;
        argsForKernel[idx++] = indexTensor;
        argsForKernel[idx++] = normalizedAxis;
        argsForKernel[idx++] = source.shape().rank();
        argsForKernel[idx++] = indexTensor.shape().rank();
        argsForKernel[idx++] = output.shape().rank();
        argsForKernel[idx++] = source.shape().size();
        argsForKernel[idx++] = indexTensor.shape().size();
        argsForKernel[idx++] = output.shape().size();
        for (int i = 0; i < source.shape().rank(); i++) {
            argsForKernel[idx++] = source.shape().flatAt(i);
        }
        for (int i = 0; i < indexTensor.shape().rank(); i++) {
            argsForKernel[idx++] = indexTensor.shape().flatAt(i);
        }
        for (int i = 0; i < output.shape().rank(); i++) {
            argsForKernel[idx++] = output.shape().flatAt(i);
        }
        launch(kernelName, argsForKernel);
        return output;
    }

    private static int[] normalizeAxes(int rank, int firstAxis, int... otherAxes) {
        int[] raw = new int[otherAxes.length + 1];
        raw[0] = firstAxis;
        System.arraycopy(otherAxes, 0, raw, 1, otherAxes.length);
        boolean[] seen = new boolean[rank];
        int uniqueCount = 0;
        for (int value : raw) {
            int normalized = Util.wrapAround(value, rank);
            if (!seen[normalized]) {
                seen[normalized] = true;
                uniqueCount++;
            }
        }
        int[] axes = new int[uniqueCount];
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            if (seen[i]) {
                axes[idx++] = i;
            }
        }
        return axes;
    }

    private static Shape reduceShape(Shape inputShape, int[] axes, boolean keepDims) {
        boolean[] reduced = new boolean[inputShape.rank()];
        for (int axis : axes) {
            reduced[axis] = true;
        }
        if (keepDims) {
            long[] dims = new long[inputShape.rank()];
            for (int i = 0; i < inputShape.rank(); i++) {
                dims[i] = reduced[i] ? 1 : inputShape.flatAt(i);
            }
            return Shape.flat(dims);
        }
        long[] dims = new long[inputShape.rank() - axes.length];
        int out = 0;
        for (int i = 0; i < inputShape.rank(); i++) {
            if (!reduced[i]) {
                dims[out++] = inputShape.flatAt(i);
            }
        }
        return Shape.flat(dims);
    }

    private Tensor ensureContiguous(Tensor tensor, String op) {
        MemoryView<?> view = tensor.materialize();
        if (view.layout().isSuffixContiguous(0)) {
            return tensor;
        }
        throw new UnsupportedOperationException(
                "Load-only eager " + op + " kernels require contiguous inputs");
    }

    private Tensor allocate(DataType dataType, Shape shape) {
        return Tensor.of(
                MemoryView.of(
                        runtime.memoryDomain().memoryAllocator().allocateMemory(dataType, shape),
                        dataType,
                        Layout.rowMajor(shape)));
    }

    private void launch(String kernelName, Object... args) {
        KernelExecutable executable =
                runtime.loadRegisteredBinaryExecutable(kernelName)
                        .orElseThrow(
                                () ->
                                        new UnsupportedOperationException(
                                                "Missing bundled eager kernel: "
                                                        + kernelName
                                                        + " on "
                                                        + runtime.device()));
        KernelArgs kernelArgs = KernelArgs.fromVarargs(args);
        executable.launch(
                LaunchConfig.auto(), kernelArgs, new ExecutionStream(runtime.device(), null, true));
    }
}

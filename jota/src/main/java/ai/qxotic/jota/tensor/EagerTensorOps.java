package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.impl.ViewTransforms;
import ai.qxotic.jota.ir.tir.ViewKind;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Objects;

public class EagerTensorOps implements TensorOps {

    private final MemoryDomain<?> domain;

    public EagerTensorOps(MemoryDomain<?> domain) {
        this.domain = Objects.requireNonNull(domain);
    }

    @Override
    public MemoryDomain<?> memoryDomain() {
        return domain;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.ADD);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.SUBTRACT);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MULTIPLY);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.DIVIDE);
    }

    @Override
    public Tensor min(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MIN);
    }

    @Override
    public Tensor max(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MAX);
    }

    @Override
    public Tensor negate(Tensor x) {
        return unaryOp(x, UnaryOp.NEGATE);
    }

    @Override
    public Tensor abs(Tensor x) {
        return unaryOp(x, UnaryOp.ABS);
    }

    @Override
    public Tensor exp(Tensor x) {
        return unaryOp(x, UnaryOp.EXP);
    }

    @Override
    public Tensor log(Tensor x) {
        return unaryOp(x, UnaryOp.LOG);
    }

    @Override
    public Tensor sqrt(Tensor x) {
        assertFloatingPoint(x.dataType(), "sqrt");
        return unaryOp(x, UnaryOp.SQRT);
    }

    @Override
    public Tensor sin(Tensor x) {
        assertFloatingPoint(x.dataType(), "sin");
        return unaryOp(x, UnaryOp.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        assertFloatingPoint(x.dataType(), "cos");
        return unaryOp(x, UnaryOp.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        assertFloatingPoint(x.dataType(), "tanh");
        return unaryOp(x, UnaryOp.TANH);
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOp.RECIPROCAL);
    }

    @Override
    public Tensor to(Tensor x, Device device) {
        Objects.requireNonNull(device, "device");
        if (x.device().equals(device)) {
            return x;
        }
        MemoryView<?> sourceView = x.materialize();
        MemoryDomain<?> sourceContext = Environment.current().runtimeFor(x.device()).memoryDomain();
        MemoryDomain<?> targetContext = Environment.current().runtimeFor(device).memoryDomain();
        if (!targetContext.supportsDataType(sourceView.dataType())) {
            throw new IllegalArgumentException(
                    "Target domain does not support data type: " + sourceView.dataType());
        }
        if (!sourceContext.device().equals(targetContext.device()) && !sourceView.isContiguous()) {
            throw new IllegalArgumentException(
                    "Target backend cannot preserve layout; call x.contiguous().to(device)");
        }
        OutputBufferSpec outputSpec = computeOutputSpec(sourceView.layout(), sourceView.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        targetContext.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        sourceView.dataType(),
                        sourceView.layout());
        copyBetweenContexts(sourceContext, sourceView, targetContext, targetView);
        return Tensor.of(targetView);
    }

    @Override
    public Tensor contiguous(Tensor x) {
        MemoryView<?> sourceView = x.materialize();
        // Check for row-major contiguity (isSuffixContiguous(0)), not just spanning a contiguous
        // range
        // A tensor can span a contiguous range but have non-row-major strides (e.g., transposed)
        if (sourceView.layout().isSuffixContiguous(0)) {
            return x;
        }
        boolean primitive =
                sourceView.dataType() == DataType.BOOL
                        || sourceView.dataType() == DataType.I8
                        || sourceView.dataType() == DataType.I16
                        || sourceView.dataType() == DataType.I32
                        || sourceView.dataType() == DataType.I64
                        || sourceView.dataType() == DataType.FP16
                        || sourceView.dataType() == DataType.BF16
                        || sourceView.dataType() == DataType.FP32
                        || sourceView.dataType() == DataType.FP64;
        if (!primitive) {
            throw new IllegalArgumentException(
                    "contiguous requires primitive data type, got " + sourceView.dataType());
        }
        Layout layout = Layout.rowMajor(sourceView.layout().shape());
        OutputBufferSpec outputSpec = computeOutputSpec(layout, sourceView.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        domain.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        sourceView.dataType(),
                        layout);
        copyBetweenContexts(domain, sourceView, domain, targetView);
        return Tensor.of(targetView);
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        assertIntegral(x.dataType(), "bitwiseNot");
        return unaryOp(x, UnaryOp.BITWISE_NOT);
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        assertIntegral(a.dataType(), "bitwiseAnd");
        assertIntegral(b.dataType(), "bitwiseAnd");
        return binaryOp(a, b, BinaryOp.BITWISE_AND);
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        assertIntegral(a.dataType(), "bitwiseOr");
        assertIntegral(b.dataType(), "bitwiseOr");
        return binaryOp(a, b, BinaryOp.BITWISE_OR);
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        assertIntegral(a.dataType(), "bitwiseXor");
        assertIntegral(b.dataType(), "bitwiseXor");
        return binaryOp(a, b, BinaryOp.BITWISE_XOR);
    }

    @Override
    public Tensor logicalNot(Tensor x) {
        return unaryOp(x, UnaryOp.LOGICAL_NOT);
    }

    @Override
    public Tensor logicalAnd(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va && vb);
    }

    @Override
    public Tensor logicalOr(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va || vb);
    }

    @Override
    public Tensor logicalXor(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va ^ vb);
    }

    @FunctionalInterface
    private interface LogicalBinaryFunc {
        boolean apply(boolean a, boolean b);
    }

    private Tensor logicalBinaryOp(Tensor a, Tensor b, LogicalBinaryFunc func) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        if (viewA.dataType() != DataType.BOOL || viewB.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Logical ops require BOOL tensors");
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(DataType.BOOL, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            boolean va = access.readByte(memA, offA) != 0;
            boolean vb = access.readByte(memB, offB) != 0;
            access.writeByte(memOut, offOut, (byte) (func.apply(va, vb) ? 1 : 0));
        }
        return Tensor.of(output);
    }

    @Override
    public Tensor equal(Tensor a, Tensor b) {
        return compareOp(a, b, BinaryOp.EQUAL);
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return compareOp(a, b, BinaryOp.LESS_THAN);
    }

    private Tensor compareOp(Tensor a, Tensor b, BinaryOp op) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        DataType dtype = viewA.dataType();
        if (dtype != viewB.dataType()) {
            throw new IllegalArgumentException(
                    "Type mismatch: " + dtype + " vs " + viewB.dataType());
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(DataType.BOOL, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            boolean result = applyCompareOp(access, memA, offA, memB, offB, dtype, op);
            access.writeByte(memOut, offOut, (byte) (result ? 1 : 0));
        }
        return Tensor.of(output);
    }

    private boolean applyCompareOp(
            MemoryAccess<Object> access,
            Memory<Object> memA,
            long offA,
            Memory<Object> memB,
            long offB,
            DataType dtype,
            BinaryOp op) {
        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte a = access.readByte(memA, offA);
            byte b = access.readByte(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I16) {
            short a = access.readShort(memA, offA);
            short b = access.readShort(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I32) {
            int a = access.readInt(memA, offA);
            int b = access.readInt(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I64) {
            long a = access.readLong(memA, offA);
            long b = access.readLong(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.FP32) {
            float a = access.readFloat(memA, offA);
            float b = access.readFloat(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.FP64) {
            double a = access.readDouble(memA, offA);
            double b = access.readDouble(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        assertBool(condition.dataType(), "where condition");
        assert trueValue.dataType() == falseValue.dataType()
                : "where requires true and false values to have the same type, got "
                        + trueValue.dataType()
                        + " and "
                        + falseValue.dataType();
        throw new UnsupportedOperationException("Generic where op dispatch not yet implemented");
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        throw new UnsupportedOperationException("mean() not yet implemented");
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor gather(Tensor input, Tensor indices, int axis) {
        MemoryView<?> inputView = input.materialize();
        MemoryView<?> indicesView = indices.materialize();

        // Validate indices is integral type
        DataType indicesType = indicesView.dataType();
        if (!indicesType.isIntegral()) {
            throw new IllegalArgumentException(
                    "Gather indices must be integral type, got " + indicesType);
        }

        int inputRank = inputView.shape().rank();
        int normalizedAxis = Util.wrapAround(axis, inputRank);

        // Compute output shape
        Shape outputShape =
                ai.qxotic.jota.ir.tir.GatherOp.computeOutputShape(
                        inputView.shape(), indicesView.shape(), axis);

        DataType dtype = inputView.dataType();
        Layout outputLayout = Layout.rowMajor(outputShape);
        MemoryView<?> output = allocate(dtype, outputShape);

        MemoryAccess<Object> access = requireMemoryAccess();

        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            long[] outCoord = Indexing.linearToCoord(outputLayout.shape(), outIdx);

            // Build input coordinate from output coordinate
            // Output shape: [input[0:axis], indices..., input[axis+1:]]
            long[] inCoord = new long[inputRank];
            int indicesRank = indicesView.shape().rank();

            // Fill dimensions before axis: copy directly from output
            for (int i = 0; i < normalizedAxis; i++) {
                inCoord[i] = outCoord[i];
            }

            // At axis position: lookup in indices tensor
            long[] idxCoord = new long[indicesRank];
            for (int j = 0; j < indicesRank; j++) {
                idxCoord[j] = outCoord[normalizedAxis + j];
            }
            long idxOffset = Indexing.coordToOffset(indicesView, idxCoord);
            inCoord[normalizedAxis] = readIndexValue(access, indicesView, idxOffset, indicesType);

            // Fill dimensions after axis: copy from output after indices dimensions
            for (int i = normalizedAxis + 1; i < inputRank; i++) {
                inCoord[i] = outCoord[normalizedAxis + indicesRank + (i - normalizedAxis - 1)];
            }

            // Read from input and write to output
            long inOffset = Indexing.coordToOffset(inputView, inCoord);
            long outOffset = Indexing.linearToOffset(output, outIdx);

            copyValue(access, inputView, inOffset, output, outOffset, dtype);
        }

        return Tensor.of(output);
    }

    private long readIndexValue(
            MemoryAccess<Object> access, MemoryView<?> indices, long offset, DataType dtype) {
        @SuppressWarnings("unchecked")
        Memory<Object> mem = (Memory<Object>) indices.memory();
        if (dtype == DataType.I8) {
            return access.readByte(mem, offset);
        } else if (dtype == DataType.I16) {
            return access.readShort(mem, offset);
        } else if (dtype == DataType.I32) {
            return access.readInt(mem, offset);
        } else if (dtype == DataType.I64) {
            return access.readLong(mem, offset);
        } else {
            throw new UnsupportedOperationException("Unsupported index data type: " + dtype);
        }
    }

    private void copyValue(
            MemoryAccess<Object> access,
            MemoryView<?> src,
            long srcOffset,
            MemoryView<?> dst,
            long dstOffset,
            DataType dtype) {
        @SuppressWarnings("unchecked")
        Memory<Object> srcMem = (Memory<Object>) src.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> dstMem = (Memory<Object>) dst.memory();

        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte v = access.readByte(srcMem, srcOffset);
            access.writeByte(dstMem, dstOffset, v);
        } else if (dtype == DataType.I16) {
            short v = access.readShort(srcMem, srcOffset);
            access.writeShort(dstMem, dstOffset, v);
        } else if (dtype == DataType.I32) {
            int v = access.readInt(srcMem, srcOffset);
            access.writeInt(dstMem, dstOffset, v);
        } else if (dtype == DataType.I64) {
            long v = access.readLong(srcMem, srcOffset);
            access.writeLong(dstMem, dstOffset, v);
        } else if (dtype == DataType.FP32) {
            float v = access.readFloat(srcMem, srcOffset);
            access.writeFloat(dstMem, dstOffset, v);
        } else if (dtype == DataType.FP64) {
            double v = access.readDouble(srcMem, srcOffset);
            access.writeDouble(dstMem, dstOffset, v);
        } else if (dtype == DataType.FP16) {
            short v = access.readShort(srcMem, srcOffset);
            access.writeShort(dstMem, dstOffset, v);
        } else if (dtype == DataType.BF16) {
            short v = access.readShort(srcMem, srcOffset);
            access.writeShort(dstMem, dstOffset, v);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        MemoryView<?> leftView = a.materialize();
        MemoryView<?> rightView = b.materialize();
        if (leftView.shape().rank() != 2 || rightView.shape().rank() != 2) {
            throw new IllegalArgumentException(
                    "matmul requires rank-2 tensors, got "
                            + leftView.shape()
                            + " and "
                            + rightView.shape());
        }

        long m = leftView.shape().flatAt(0);
        long k = leftView.shape().flatAt(1);
        long rightK = rightView.shape().flatAt(0);
        long n = rightView.shape().flatAt(1);
        if (k != rightK) {
            throw new IllegalArgumentException(
                    "matmul inner dimensions must match, got " + k + " and " + rightK);
        }

        DataType dtype = leftView.dataType();
        if (dtype != rightView.dataType()) {
            throw new IllegalArgumentException(
                    "matmul requires matching input dtypes, got "
                            + dtype
                            + " and "
                            + rightView.dataType());
        }

        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, Shape.of(m, n));
        @SuppressWarnings("unchecked")
        Memory<Object> leftMem = (Memory<Object>) leftView.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> rightMem = (Memory<Object>) rightView.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> outMem = (Memory<Object>) output.memory();

        long[] leftCoord = new long[2];
        long[] rightCoord = new long[2];
        long[] outCoord = new long[2];
        for (long i = 0; i < m; i++) {
            leftCoord[0] = i;
            outCoord[0] = i;
            for (long j = 0; j < n; j++) {
                rightCoord[1] = j;
                outCoord[1] = j;
                double sum = 0.0;
                for (long kk = 0; kk < k; kk++) {
                    leftCoord[1] = kk;
                    rightCoord[0] = kk;
                    long leftOffset = Indexing.coordToOffset(leftView, leftCoord);
                    long rightOffset = Indexing.coordToOffset(rightView, rightCoord);
                    double leftValue = readAsDouble(access, leftMem, leftOffset, dtype);
                    double rightValue = readAsDouble(access, rightMem, rightOffset, dtype);
                    sum += leftValue * rightValue;
                }
                long outOffset = Indexing.coordToOffset(output, outCoord);
                writeFromDouble(access, outMem, outOffset, dtype, sum);
            }
        }

        return Tensor.of(output);
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        MemoryView<?> leftView = a.materialize();
        MemoryView<?> rightView = b.materialize();
        if (leftView.shape().rank() != 3 || rightView.shape().rank() != 3) {
            throw new IllegalArgumentException(
                    "batchedMatmul requires rank-3 tensors, got "
                            + leftView.shape()
                            + " and "
                            + rightView.shape());
        }

        long batch = leftView.shape().flatAt(0);
        long rightBatch = rightView.shape().flatAt(0);
        long m = leftView.shape().flatAt(1);
        long k = leftView.shape().flatAt(2);
        long rightK = rightView.shape().flatAt(1);
        long n = rightView.shape().flatAt(2);
        if (batch != rightBatch) {
            throw new IllegalArgumentException(
                    "batchedMatmul batch dimensions must match, got "
                            + batch
                            + " and "
                            + rightBatch);
        }
        if (k != rightK) {
            throw new IllegalArgumentException(
                    "batchedMatmul inner dimensions must match, got " + k + " and " + rightK);
        }

        DataType dtype = leftView.dataType();
        if (dtype != rightView.dataType()) {
            throw new IllegalArgumentException(
                    "batchedMatmul requires matching input dtypes, got "
                            + dtype
                            + " and "
                            + rightView.dataType());
        }

        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, Shape.of(batch, m, n));
        @SuppressWarnings("unchecked")
        Memory<Object> leftMem = (Memory<Object>) leftView.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> rightMem = (Memory<Object>) rightView.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> outMem = (Memory<Object>) output.memory();

        long[] leftCoord = new long[3];
        long[] rightCoord = new long[3];
        long[] outCoord = new long[3];
        for (long bIdx = 0; bIdx < batch; bIdx++) {
            leftCoord[0] = bIdx;
            rightCoord[0] = bIdx;
            outCoord[0] = bIdx;
            for (long i = 0; i < m; i++) {
                leftCoord[1] = i;
                outCoord[1] = i;
                for (long j = 0; j < n; j++) {
                    rightCoord[2] = j;
                    outCoord[2] = j;
                    double sum = 0.0;
                    for (long kk = 0; kk < k; kk++) {
                        leftCoord[2] = kk;
                        rightCoord[1] = kk;
                        long leftOffset = Indexing.coordToOffset(leftView, leftCoord);
                        long rightOffset = Indexing.coordToOffset(rightView, rightCoord);
                        double leftValue = readAsDouble(access, leftMem, leftOffset, dtype);
                        double rightValue = readAsDouble(access, rightMem, rightOffset, dtype);
                        sum += leftValue * rightValue;
                    }
                    long outOffset = Indexing.coordToOffset(output, outCoord);
                    writeFromDouble(access, outMem, outOffset, dtype, sum);
                }
            }
        }

        return Tensor.of(output);
    }

    @Override
    public Tensor viewTransform(Tensor x, ViewTransforms.ViewTransformSpec spec) {
        // Preserve lazy ConstantComputation for broadcasts/reshapes - don't materialize
        // This is critical for IR tracing to recognize constants
        if (x.computation().orElse(null) instanceof ConstantComputation constComp
                && !x.isMaterialized()) {
            Shape newShape = spec.layout().shape();
            // For constants, we can just create a new lazy constant with the new shape
            // The strides will be all zeros (broadcast semantics)
            return Tensor.lazy(
                    new ConstantComputation(
                            constComp.rawBits(),
                            constComp.dataType(),
                            newShape,
                            constComp.device()),
                    constComp.dataType(),
                    spec.layout(),
                    x.device());
        }

        // Eager path: if lazy indexing is needed, force contiguous copy first
        if (spec.needsLazyIndexing()) {
            Tensor contiguousTensor = contiguous(x);
            // Recompute spec with contiguous layout
            ViewTransforms.ViewTransformSpec newSpec =
                    switch (spec.kind()) {
                        case ViewKind.Reshape reshape ->
                                ViewTransforms.view(contiguousTensor.layout(), reshape.toShape());
                        case ViewKind.Broadcast broadcast ->
                                ViewTransforms.broadcast(
                                        contiguousTensor.layout(), broadcast.toShape());
                        case ViewKind.Expand expand ->
                                ViewTransforms.expand(contiguousTensor.layout(), expand.toShape());
                        case ViewKind.Transpose transpose ->
                                ViewTransforms.permute(
                                        contiguousTensor.layout(), transpose.permutation());
                        case ViewKind.Slice slice ->
                                ViewTransforms.slice(
                                        contiguousTensor.layout(),
                                        x.dataType(),
                                        slice.axis(),
                                        slice.start(),
                                        slice.start() + spec.layout().shape().flatAt(slice.axis()),
                                        slice.step());
                    };
            return viewTransform(contiguousTensor, newSpec);
        }

        MemoryView<?> sourceView = x.materialize();
        MemoryView<?> transformed =
                MemoryView.of(
                        sourceView.memory(),
                        sourceView.byteOffset() + spec.byteOffsetDelta(),
                        sourceView.dataType(),
                        spec.layout());
        return Tensor.of(transformed);
    }

    @Override
    public Tensor reshape(Tensor x, Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(x.layout(), newShape);
        return viewTransform(x, spec);
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        MemoryView<?> view = x.materialize();
        if (view.dataType() == targetType) {
            return x;
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(targetType, view.shape());
        long size = view.shape().size();
        @SuppressWarnings("unchecked")
        Memory<Object> srcMemory = (Memory<Object>) view.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> dstMemory = (Memory<Object>) output.memory();
        DataType srcType = view.dataType();
        for (long i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(view, i);
            long dstOffset = Indexing.linearToOffset(output, i);
            double value = readAsDouble(access, srcMemory, srcOffset, srcType);
            writeFromDouble(access, dstMemory, dstOffset, targetType, value);
        }
        return Tensor.of(output);
    }

    private double readAsDouble(
            MemoryAccess<Object> access, Memory<Object> memory, long offset, DataType type) {
        if (type == DataType.BOOL || type == DataType.I8) {
            return access.readByte(memory, offset);
        } else if (type == DataType.I16 || type == DataType.FP16 || type == DataType.BF16) {
            return access.readShort(memory, offset);
        } else if (type == DataType.I32) {
            return access.readInt(memory, offset);
        } else if (type == DataType.FP32) {
            return access.readFloat(memory, offset);
        } else if (type == DataType.I64) {
            return access.readLong(memory, offset);
        } else if (type == DataType.FP64) {
            return access.readDouble(memory, offset);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + type);
        }
    }

    private void writeFromDouble(
            MemoryAccess<Object> access,
            Memory<Object> memory,
            long offset,
            DataType type,
            double value) {
        if (type == DataType.BOOL) {
            access.writeByte(memory, offset, (byte) (value != 0 ? 1 : 0));
        } else if (type == DataType.I8) {
            access.writeByte(memory, offset, (byte) value);
        } else if (type == DataType.I16) {
            access.writeShort(memory, offset, (short) value);
        } else if (type == DataType.I32) {
            access.writeInt(memory, offset, (int) value);
        } else if (type == DataType.FP32) {
            access.writeFloat(memory, offset, (float) value);
        } else if (type == DataType.I64) {
            access.writeLong(memory, offset, (long) value);
        } else if (type == DataType.FP64) {
            access.writeDouble(memory, offset, value);
        } else if (type == DataType.FP16 || type == DataType.BF16) {
            access.writeShort(memory, offset, (short) Float.floatToFloat16((float) value));
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + type);
        }
    }

    private MemoryView<?> allocate(DataType dtype, Shape shape) {
        Layout layout = Layout.rowMajor(shape);
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(dtype.byteSizeFor(shape)), dtype, layout);
    }

    private Tensor unaryOp(Tensor x, UnaryOp op) {
        MemoryView<?> view = x.materialize();
        DataType dtype = view.dataType();
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, view.shape());
        long size = view.shape().size();
        @SuppressWarnings("unchecked")
        Memory<Object> srcMemory = (Memory<Object>) view.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> dstMemory = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(view, i);
            long dstOffset = Indexing.linearToOffset(output, i);
            applyUnaryOp(access, srcMemory, srcOffset, dstMemory, dstOffset, dtype, op);
        }
        return Tensor.of(output);
    }

    private void applyUnaryOp(
            MemoryAccess<Object> access,
            Memory<Object> srcMem,
            long srcOff,
            Memory<Object> dstMem,
            long dstOff,
            DataType dtype,
            UnaryOp op) {
        if (dtype == DataType.BOOL) {
            byte v = access.readByte(srcMem, srcOff);
            access.writeByte(dstMem, dstOff, ConstantFolder.evalBool(v != 0, op));
        } else if (dtype == DataType.I8) {
            byte v = access.readByte(srcMem, srcOff);
            access.writeByte(dstMem, dstOff, ConstantFolder.evalByte(v, op));
        } else if (dtype == DataType.I16) {
            short v = access.readShort(srcMem, srcOff);
            access.writeShort(dstMem, dstOff, ConstantFolder.evalShort(v, op));
        } else if (dtype == DataType.I32) {
            int v = access.readInt(srcMem, srcOff);
            access.writeInt(dstMem, dstOff, ConstantFolder.evalInt(v, op));
        } else if (dtype == DataType.I64) {
            long v = access.readLong(srcMem, srcOff);
            access.writeLong(dstMem, dstOff, ConstantFolder.evalLong(v, op));
        } else if (dtype == DataType.FP32) {
            float v = access.readFloat(srcMem, srcOff);
            access.writeFloat(dstMem, dstOff, ConstantFolder.evalFloat(v, op));
        } else if (dtype == DataType.FP64) {
            double v = access.readDouble(srcMem, srcOff);
            access.writeDouble(dstMem, dstOff, ConstantFolder.evalDouble(v, op));
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    private MemoryAccess<Object> requireMemoryAccess() {
        MemoryAccess<?> access = domain.directAccess();
        if (access == null) {
            throw new IllegalStateException("Eager ops require MemoryAccess");
        }
        @SuppressWarnings("unchecked")
        MemoryAccess<Object> cast = (MemoryAccess<Object>) access;
        return cast;
    }

    private static void copyBetweenContexts(
            MemoryDomain<?> sourceContext,
            MemoryView<?> sourceView,
            MemoryDomain<?> targetContext,
            MemoryView<?> targetView) {
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> source = (MemoryDomain<Object>) sourceContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> src = (MemoryView<Object>) sourceView;
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> target = (MemoryDomain<Object>) targetContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> dst = (MemoryView<Object>) targetView;
        MemoryDomain.copy(source, src, target, dst);
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

    private record OutputBufferSpec(long byteOffset, long byteSize) {}

    private static void assertIntegral(DataType dataType, String opName) {
        assert dataType.isIntegral() && dataType != DataType.BOOL
                : opName + " requires integral data type, got " + dataType;
    }

    private static void assertFloatingPoint(DataType dataType, String opName) {
        assert dataType.isFloatingPoint()
                : opName + " requires floating-point data type, got " + dataType;
    }

    private static void assertBool(DataType dataType, String opName) {
        assert dataType == DataType.BOOL : opName + " requires BOOL data type, got " + dataType;
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOp op) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        DataType dtype = viewA.dataType();
        if (dtype != viewB.dataType()) {
            throw new IllegalArgumentException(
                    "Type mismatch: " + dtype + " vs " + viewB.dataType());
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            applyBinaryOp(access, memA, offA, memB, offB, memOut, offOut, dtype, op);
        }
        return Tensor.of(output);
    }

    private void applyBinaryOp(
            MemoryAccess<Object> access,
            Memory<Object> memA,
            long offA,
            Memory<Object> memB,
            long offB,
            Memory<Object> memOut,
            long offOut,
            DataType dtype,
            BinaryOp op) {
        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte a = access.readByte(memA, offA);
            byte b = access.readByte(memB, offB);
            access.writeByte(memOut, offOut, ConstantFolder.evalByte(a, b, op));
        } else if (dtype == DataType.I16) {
            short a = access.readShort(memA, offA);
            short b = access.readShort(memB, offB);
            access.writeShort(memOut, offOut, ConstantFolder.evalShort(a, b, op));
        } else if (dtype == DataType.I32) {
            int a = access.readInt(memA, offA);
            int b = access.readInt(memB, offB);
            access.writeInt(memOut, offOut, ConstantFolder.evalInt(a, b, op));
        } else if (dtype == DataType.I64) {
            long a = access.readLong(memA, offA);
            long b = access.readLong(memB, offB);
            access.writeLong(memOut, offOut, ConstantFolder.evalLong(a, b, op));
        } else if (dtype == DataType.FP32) {
            float a = access.readFloat(memA, offA);
            float b = access.readFloat(memB, offB);
            access.writeFloat(memOut, offOut, ConstantFolder.evalFloat(a, b, op));
        } else if (dtype == DataType.FP64) {
            double a = access.readDouble(memA, offA);
            double b = access.readDouble(memB, offB);
            access.writeDouble(memOut, offOut, ConstantFolder.evalDouble(a, b, op));
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    private Tensor scalarOp(Tensor a, Number scalar, BinaryOp op) {
        throw new UnsupportedOperationException("Generic scalar op dispatch not yet implemented");
    }
}

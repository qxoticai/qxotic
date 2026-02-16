package ai.qxotic.jota.tensor;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.impl.ViewTransforms;
import ai.qxotic.jota.ir.tir.GatherOp;
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
        TensorTypeSemantics.requireFloatingPoint(x.dataType(), "sqrt");
        return unaryOp(x, UnaryOp.SQRT);
    }

    @Override
    public Tensor sin(Tensor x) {
        TensorTypeSemantics.requireFloatingPoint(x.dataType(), "sin");
        return unaryOp(x, UnaryOp.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        TensorTypeSemantics.requireFloatingPoint(x.dataType(), "cos");
        return unaryOp(x, UnaryOp.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        TensorTypeSemantics.requireFloatingPoint(x.dataType(), "tanh");
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
        TensorTypeSemantics.requireIntegral(x.dataType(), "bitwiseNot");
        return unaryOp(x, UnaryOp.BITWISE_NOT);
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOp.BITWISE_AND, "bitwiseAnd");
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOp.BITWISE_OR, "bitwiseOr");
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOp.BITWISE_XOR, "bitwiseXor");
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
        TensorTypeSemantics.requireBooleanPair(viewA.dataType(), viewB.dataType(), "logical op");
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
        DataType dtype =
                TensorTypeSemantics.promoteForComparison(
                        viewA.dataType(), viewB.dataType(), op.name());
        MemoryView<?> left = castIfNeeded(viewA, dtype);
        MemoryView<?> right = castIfNeeded(viewB, dtype);
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(DataType.BOOL, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) left.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) right.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(left, i);
            long offB = Indexing.linearToOffset(right, i);
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
        } else if (dtype == DataType.FP16) {
            float a = Float.float16ToFloat(access.readShort(memA, offA));
            float b = Float.float16ToFloat(access.readShort(memB, offB));
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.BF16) {
            float a = BFloat16.toFloat(access.readShort(memA, offA));
            float b = BFloat16.toFloat(access.readShort(memB, offB));
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
        TensorTypeSemantics.requireBool(condition.dataType(), "where condition");
        if (trueValue.dataType() != falseValue.dataType()) {
            throw new IllegalArgumentException(
                    "where requires true and false values to have the same type, got "
                            + trueValue.dataType()
                            + " and "
                            + falseValue.dataType());
        }

        MemoryView<?> conditionView = condition.materialize();
        MemoryView<?> trueView = trueValue.materialize();
        MemoryView<?> falseView = falseValue.materialize();

        Shape outputShape =
                TensorSemantics.resolveWhereShape(
                        conditionView.shape(), trueView.shape(), falseView.shape());
        DataType outputType = trueView.dataType();
        MemoryView<?> output = allocate(outputType, outputShape);
        MemoryAccess<Object> access = requireMemoryAccess();

        long outputSize = outputShape.size();
        for (long outLinear = 0; outLinear < outputSize; outLinear++) {
            long[] outCoord = Indexing.linearToCoord(outputShape, outLinear);
            long conditionOffset = coordForWhereInput(conditionView, outCoord);
            long trueOffset = coordForWhereInput(trueView, outCoord);
            long falseOffset = coordForWhereInput(falseView, outCoord);
            long outOffset = Indexing.linearToOffset(output, outLinear);

            @SuppressWarnings("unchecked")
            Memory<Object> conditionMemory = (Memory<Object>) conditionView.memory();
            boolean cond = access.readByte(conditionMemory, conditionOffset) != 0;
            if (cond) {
                copyValue(access, trueView, trueOffset, output, outOffset, outputType);
            } else {
                copyValue(access, falseView, falseOffset, output, outOffset, outputType);
            }
        }

        return Tensor.of(output);
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduce(x, ReductionOpKind.SUM, accumulatorType, keepDims, _axis, _axes);
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduce(x, ReductionOpKind.PRODUCT, accumulatorType, keepDims, _axis, _axes);
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        DataType dtype = x.dataType();
        TensorTypeSemantics.requireFloatingPoint(dtype, "mean");
        Tensor sum = sum(x, dtype, keepDims, axis);
        int rank = x.shape().rank();
        int normalizedAxis = TensorSemantics.normalizeAxis(rank, axis);
        long count = x.shape().flatAt(normalizedAxis);
        return sum.divide(Tensor.scalar((double) count, dtype));
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduce(x, ReductionOpKind.MAX, null, keepDims, _axis, _axes);
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduce(x, ReductionOpKind.MIN, null, keepDims, _axis, _axes);
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
        int normalizedAxis = TensorSemantics.normalizeAxis(inputRank, axis);

        // Compute output shape
        Shape outputShape =
                GatherOp.computeOutputShape(
                        inputView.shape(), indicesView.shape(), normalizedAxis);

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
        } else if (type == DataType.I16) {
            return access.readShort(memory, offset);
        } else if (type == DataType.FP16) {
            return Float.float16ToFloat(access.readShort(memory, offset));
        } else if (type == DataType.BF16) {
            return BFloat16.toFloat(access.readShort(memory, offset));
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
            short bits =
                    type == DataType.FP16
                            ? Float.floatToFloat16((float) value)
                            : BFloat16.fromFloat((float) value);
            access.writeShort(memory, offset, bits);
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

    private Tensor reduce(
            Tensor x,
            ReductionOpKind kind,
            DataType accumulatorType,
            boolean keepDims,
            int _axis,
            int... _axes) {
        MemoryView<?> inputView = x.materialize();
        int[] axes = TensorSemantics.normalizeReductionAxes(inputView.shape().rank(), _axis, _axes);
        Shape outputShape = TensorSemantics.reduceShape(inputView.shape(), axes, keepDims);
        boolean[] reducedMask = TensorSemantics.reductionMask(inputView.shape().rank(), axes);
        DataType outputType =
                kind == ReductionOpKind.SUM || kind == ReductionOpKind.PRODUCT
                        ? TensorTypeSemantics.resolveReductionAccumulator(
                                inputView.dataType(), accumulatorType, kind.name().toLowerCase())
                        : inputView.dataType();
        MemoryView<?> output = allocate(outputType, outputShape);
        MemoryAccess<Object> access = requireMemoryAccess();
        long outputSize = outputShape.size();
        if (outputSize == 0) {
            return Tensor.of(output);
        }

        boolean[] initialized = new boolean[Math.toIntExact(outputSize)];
        long inputSize = inputView.shape().size();
        for (long inLinear = 0; inLinear < inputSize; inLinear++) {
            long[] inCoord = Indexing.linearToCoord(inputView.shape(), inLinear);
            long[] outCoord = TensorSemantics.projectReducedCoord(inCoord, reducedMask, keepDims);
            long outLinear = Indexing.coordToLinear(outputShape, outCoord);
            int outIndex = Math.toIntExact(outLinear);
            long inOffset = Indexing.linearToOffset(inputView, inLinear);
            long outOffset = Indexing.linearToOffset(output, outLinear);

            if (!initialized[outIndex]) {
                double inputValue = readAsDoubleFromView(access, inputView, inOffset);
                writeFromDoubleToView(access, output, outOffset, inputValue);
                initialized[outIndex] = true;
                continue;
            }

            double accValue = readAsDoubleFromView(access, output, outOffset);
            double inputValue = readAsDoubleFromView(access, inputView, inOffset);
            double nextValue =
                    switch (kind) {
                        case SUM -> accValue + inputValue;
                        case PRODUCT -> accValue * inputValue;
                        case MIN -> Math.min(accValue, inputValue);
                        case MAX -> Math.max(accValue, inputValue);
                    };
            writeFromDoubleToView(access, output, outOffset, nextValue);
        }

        return Tensor.of(output);
    }

    private double readAsDoubleFromView(MemoryAccess<Object> access, MemoryView<?> view, long offset) {
        @SuppressWarnings("unchecked")
        Memory<Object> memory = (Memory<Object>) view.memory();
        return readAsDouble(access, memory, offset, view.dataType());
    }

    private void writeFromDoubleToView(
            MemoryAccess<Object> access, MemoryView<?> view, long offset, double value) {
        @SuppressWarnings("unchecked")
        Memory<Object> memory = (Memory<Object>) view.memory();
        writeFromDouble(access, memory, offset, view.dataType(), value);
    }

    private static long coordForWhereInput(MemoryView<?> view, long[] outCoord) {
        if (view.shape().isScalar()) {
            return Indexing.linearToOffset(view, 0);
        }
        if (view.shape().rank() != outCoord.length) {
            throw new IllegalArgumentException(
                    "where shape mismatch: expected rank "
                            + outCoord.length
                            + " but got "
                            + view.shape().rank());
        }
        return Indexing.coordToOffset(view, outCoord);
    }

    private enum ReductionOpKind {
        SUM,
        PRODUCT,
        MIN,
        MAX
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOp op) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        DataType dtype =
                TensorTypeSemantics.promoteForArithmetic(
                        viewA.dataType(), viewB.dataType(), op.name());
        MemoryView<?> left = castIfNeeded(viewA, dtype);
        MemoryView<?> right = castIfNeeded(viewB, dtype);
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) left.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) right.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(left, i);
            long offB = Indexing.linearToOffset(right, i);
            long offOut = Indexing.linearToOffset(output, i);
            applyBinaryOp(access, memA, offA, memB, offB, memOut, offOut, dtype, op);
        }
        return Tensor.of(output);
    }

    private Tensor bitwiseBinaryOp(Tensor a, Tensor b, BinaryOp op, String opName) {
        TensorTypeSemantics.requireSameIntegralType(a.dataType(), b.dataType(), opName);
        return binaryOpSameType(a, b, op);
    }

    private Tensor binaryOpSameType(Tensor a, Tensor b, BinaryOp op) {
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

    private MemoryView<?> castIfNeeded(MemoryView<?> view, DataType targetType) {
        if (view.dataType() == targetType) {
            return view;
        }
        return cast(Tensor.of(view), targetType).materialize();
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
        } else if (dtype == DataType.FP16) {
            float a = Float.float16ToFloat(access.readShort(memA, offA));
            float b = Float.float16ToFloat(access.readShort(memB, offB));
            access.writeShort(memOut, offOut, Float.floatToFloat16(ConstantFolder.evalFloat(a, b, op)));
        } else if (dtype == DataType.BF16) {
            float a = BFloat16.toFloat(access.readShort(memA, offA));
            float b = BFloat16.toFloat(access.readShort(memB, offB));
            access.writeShort(memOut, offOut, BFloat16.fromFloat(ConstantFolder.evalFloat(a, b, op)));
        } else if (dtype == DataType.FP64) {
            double a = access.readDouble(memA, offA);
            double b = access.readDouble(memB, offB);
            access.writeDouble(memOut, offOut, ConstantFolder.evalDouble(a, b, op));
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }
}

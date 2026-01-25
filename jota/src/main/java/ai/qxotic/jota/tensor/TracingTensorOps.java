package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.TypeRules;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

final class TracingTensorOps implements TensorOps {

    @Override
    public MemoryContext<?> context() {
        throw new UnsupportedOperationException("Tracing ops do not expose a memory context");
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
        return unaryOp(x, UnaryOp.SQRT);
    }

    public Tensor square(Tensor x) {
        return unaryOp(x, UnaryOp.SQUARE);
    }

    @Override
    public Tensor sin(Tensor x) {
        return unaryOp(x, UnaryOp.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        return unaryOp(x, UnaryOp.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        return unaryOp(x, UnaryOp.TANH);
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOp.RECIPROCAL);
    }

    @Override
    public Tensor to(Tensor x, Device device) {
        Objects.requireNonNull(device, "device");
        TraceTensor trace = requireTrace(x);
        if (trace.device().equals(device)) {
            return trace;
        }
        ExprNode node = new TransferNode(trace.node(), device, trace.dataType(), trace.layout());
        return new TraceTensor(node);
    }

    @Override
    public Tensor contiguous(Tensor x) {
        TraceTensor trace = requireTrace(x);
        requirePrimitive(trace.dataType(), "contiguous");
        MemoryView<?> view = trace.tryGetMaterialized().orElse(null);
        if ((view != null && view.isContiguous()) || trace.layout().shape().hasZeroElements()) {
            return trace;
        }
        Layout layout = Layout.rowMajor(trace.layout().shape());
        ExprNode node = new ContiguousNode(trace.node(), trace.dataType(), layout, trace.device());
        return new TraceTensor(node);
    }

    @Override
    public Tensor logicalNot(Tensor x) {
        TraceTensor trace = requireTrace(x);
        requireBool(trace.dataType(), "logicalNot");
        ExprNode node =
                new UnaryNode(
                        UnaryOp.LOGICAL_NOT,
                        trace.node(),
                        trace.dataType(),
                        trace.layout(),
                        trace.device());
        return new TraceTensor(node);
    }

    @Override
    public Tensor logicalAnd(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOp.LOGICAL_AND, "logicalAnd");
    }

    @Override
    public Tensor logicalOr(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOp.LOGICAL_OR, "logicalOr");
    }

    @Override
    public Tensor logicalXor(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOp.LOGICAL_XOR, "logicalXor");
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        TraceTensor trace = requireTrace(x);
        requireIntegral(trace.dataType(), "bitwiseNot");
        ExprNode node =
                new UnaryNode(
                        UnaryOp.BITWISE_NOT,
                        trace.node(),
                        trace.dataType(),
                        trace.layout(),
                        trace.device());
        return new TraceTensor(node);
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
    public Tensor equal(Tensor a, Tensor b) {
        return comparisonOp(a, b, BinaryOp.EQUAL, "equal");
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return comparisonOp(a, b, BinaryOp.LESS_THAN, "lessThan");
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        TraceTensor cond = requireTrace(condition);
        TraceTensor whenTrue = requireTrace(trueValue);
        TraceTensor whenFalse = requireTrace(falseValue);
        requireBool(cond.dataType(), "where");
        requireSameType(whenTrue, whenFalse, "where");
        Layout layout = requireCompatibleLayout(whenTrue, whenFalse);
        layout = requireCompatibleLayout(cond, whenTrue);
        Device device = requireCompatibleDevice(whenTrue, whenFalse);
        device = requireCompatibleDevice(cond, whenTrue);
        ExprNode node =
                new TernaryNode(
                        TernaryOp.WHERE,
                        cond.node(),
                        whenTrue.node(),
                        whenFalse.node(),
                        whenTrue.dataType(),
                        layout,
                        device);
        return new TraceTensor(node);
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        TraceTensor trace = requireTrace(x);
        validateAccumulator(trace.dataType(), accumulatorType, "sum");
        return reduceAxes(trace, ReductionOp.SUM, accumulatorType, keepDims, _axis, _axes);
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        TraceTensor trace = requireTrace(x);
        validateAccumulator(trace.dataType(), accumulatorType, "product");
        return reduceAxes(trace, ReductionOp.PROD, accumulatorType, keepDims, _axis, _axes);
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        throw unsupported("mean");
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.MAX, keepDims, _axis, _axes);
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.MIN, keepDims, _axis, _axes);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw unsupported("matmul");
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        throw unsupported("batchedMatmul");
    }

    @Override
    public Tensor transpose(Tensor x, int axis0, int axis1) {
        throw unsupported("transpose");
    }

    @Override
    public Tensor reshape(Tensor x, ai.qxotic.jota.Shape newShape) {
        throw unsupported("reshape");
    }

    @Override
    public Tensor view(Tensor x, ai.qxotic.jota.Shape newShape) {
        throw unsupported("view");
    }

    @Override
    public Tensor broadcast(Tensor x, ai.qxotic.jota.Shape targetShape) {
        throw unsupported("broadcast");
    }

    @Override
    public Tensor expand(Tensor x, ai.qxotic.jota.Shape targetShape) {
        throw unsupported("expand");
    }

    @Override
    public Tensor slice(Tensor x, int axis, long start, long end) {
        throw unsupported("slice");
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        Objects.requireNonNull(targetType, "targetType");
        TraceTensor trace = requireTrace(x);
        if (trace.dataType() == targetType) {
            return trace;
        }
        ExprNode node = new CastNode(trace.node(), targetType, trace.layout(), trace.device());
        return new TraceTensor(node);
    }

    private Tensor unaryOp(Tensor x, UnaryOp op) {
        TraceTensor trace = requireTrace(x);
        ExprNode node =
                new UnaryNode(op, trace.node(), trace.dataType(), trace.layout(), trace.device());
        return new TraceTensor(node);
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOp op) {
        TraceTensor left = requireTrace(a);
        TraceTensor right = requireTrace(b);
        Layout layout = requireCompatibleLayout(left, right);
        Device device = requireCompatibleDevice(left, right);
        DataType targetType = TypeRules.promote(left.dataType(), right.dataType());
        ExprNode leftNode = maybeCast(left.node(), targetType, layout, device);
        ExprNode rightNode = maybeCast(right.node(), targetType, layout, device);
        ExprNode node = new BinaryNode(op, leftNode, rightNode, targetType, layout, device);
        return new TraceTensor(node);
    }

    private Tensor booleanBinaryOp(Tensor a, Tensor b, BinaryOp op, String opName) {
        TraceTensor left = requireTrace(a);
        TraceTensor right = requireTrace(b);
        requireSameType(left, right, opName);
        requireBool(left.dataType(), opName);
        Layout layout = requireCompatibleLayout(left, right);
        Device device = requireCompatibleDevice(left, right);
        ExprNode node =
                new BinaryNode(op, left.node(), right.node(), DataType.BOOL, layout, device);
        return new TraceTensor(node);
    }

    private Tensor bitwiseBinaryOp(Tensor a, Tensor b, BinaryOp op, String opName) {
        TraceTensor left = requireTrace(a);
        TraceTensor right = requireTrace(b);
        requireSameType(left, right, opName);
        requireIntegral(left.dataType(), opName);
        Layout layout = requireCompatibleLayout(left, right);
        Device device = requireCompatibleDevice(left, right);
        ExprNode node =
                new BinaryNode(op, left.node(), right.node(), left.dataType(), layout, device);
        return new TraceTensor(node);
    }

    private Tensor comparisonOp(Tensor a, Tensor b, BinaryOp op, String opName) {
        TraceTensor left = requireTrace(a);
        TraceTensor right = requireTrace(b);
        requireSameType(left, right, opName);
        Layout layout = requireCompatibleLayout(left, right);
        Device device = requireCompatibleDevice(left, right);
        ExprNode node =
                new BinaryNode(op, left.node(), right.node(), DataType.BOOL, layout, device);
        return new TraceTensor(node);
    }

    private Tensor scalarOp(Tensor a, Number scalar, BinaryOp op) {
        Objects.requireNonNull(scalar, "scalar");
        TraceTensor left = requireTrace(a);
        Layout layout = left.layout();
        Device device = left.device();
        DataType scalarType = scalarType(scalar);
        DataType targetType = TypeRules.promote(left.dataType(), scalarType);
        ExprNode leftNode = maybeCast(left.node(), targetType, layout, device);
        ExprNode scalarNode = new ScalarNode(scalar, scalarType, layout, device);
        ExprNode rightNode = maybeCast(scalarNode, targetType, layout, device);
        ExprNode node = new BinaryNode(op, leftNode, rightNode, targetType, layout, device);
        return new TraceTensor(node);
    }

    private Tensor reduceAxes(Tensor x, ReductionOp op, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, op, x.dataType(), keepDims, _axis, _axes);
    }

    private Tensor reduceAxes(
            Tensor x,
            ReductionOp op,
            DataType accumulatorType,
            boolean keepDims,
            int _axis,
            int... _axes) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(op, "op");
        TraceTensor current = requireTrace(x);
        int[] axes = normalizeAxes(current.layout().shape(), _axis, _axes);
        for (int axis : axes) {
            current = reduceAxis(current, op, accumulatorType, axis, keepDims);
        }
        return current;
    }

    private TraceTensor reduceAxis(
            TraceTensor input,
            ReductionOp op,
            DataType accumulatorType,
            int flatAxis,
            boolean keepDims) {
        Shape inputShape = input.layout().shape();
        Shape outputShape = reduceShape(inputShape, flatAxis, keepDims);
        Layout outputLayout = Layout.rowMajor(outputShape);
        ReductionNode node =
                new ReductionNode(
                        op,
                        input.node(),
                        flatAxis,
                        keepDims,
                        accumulatorType,
                        outputLayout,
                        input.device());
        return new TraceTensor(node);
    }

    private int[] normalizeAxes(Shape shape, int firstAxis, int... moreAxes) {
        int modeRank = shape.rank();
        int total = 1 + (moreAxes == null ? 0 : moreAxes.length);
        int[] axes = new int[total];
        axes[0] = Util.wrapAround(firstAxis, modeRank);
        if (moreAxes != null) {
            for (int i = 0; i < moreAxes.length; i++) {
                axes[i + 1] = Util.wrapAround(moreAxes[i], modeRank);
            }
        }
        int[] flatAxes =
                Arrays.stream(axes)
                        .distinct()
                        .flatMap(axis -> modeToFlatAxes(shape, axis))
                        .distinct()
                        .toArray();
        Arrays.sort(flatAxes);
        for (int i = 0; i < flatAxes.length / 2; i++) {
            int tmp = flatAxes[i];
            flatAxes[i] = flatAxes[flatAxes.length - 1 - i];
            flatAxes[flatAxes.length - 1 - i] = tmp;
        }
        return flatAxes;
    }

    private java.util.stream.IntStream modeToFlatAxes(Shape shape, int modeIndex) {
        int startFlat = 0;
        for (int mode = 0; mode < modeIndex; mode++) {
            startFlat += shape.modeAt(mode).flatRank();
        }
        int modeFlatRank = shape.modeAt(modeIndex).flatRank();
        return java.util.stream.IntStream.range(startFlat, startFlat + modeFlatRank);
    }

    private Shape reduceShape(Shape shape, int flatAxis, boolean keepDims) {
        long[] dims = shape.toArray();
        int rank = dims.length;
        int wrappedAxis = Util.wrapAround(flatAxis, rank);
        if (keepDims) {
            dims[wrappedAxis] = 1;
            return Shape.flat(dims);
        }
        if (rank == 1) {
            return Shape.scalar();
        }
        long[] reduced = new long[rank - 1];
        int outIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (i == wrappedAxis) {
                continue;
            }
            reduced[outIndex++] = dims[i];
        }
        return Shape.flat(reduced);
    }

    private void validateAccumulator(
            DataType inputType, DataType accumulatorType, String reductionName) {
        Objects.requireNonNull(accumulatorType, "accumulatorType");
        if (inputType == DataType.BOOL) {
            if (accumulatorType != DataType.I32 && accumulatorType != DataType.I64) {
                throw new IllegalArgumentException(
                        "BOOL "
                                + reductionName
                                + " accumulator must be I32 or I64, got "
                                + accumulatorType);
            }
            return;
        }
        if (inputType.isFloatingPoint()) {
            if (accumulatorType != DataType.FP32 && accumulatorType != DataType.FP64) {
                throw new IllegalArgumentException(
                        "Floating-point "
                                + reductionName
                                + " accumulator must be FP32 or FP64, got "
                                + accumulatorType);
            }
            return;
        }
        if (inputType.isIntegral()) {
            boolean isFloatAccumulator =
                    accumulatorType == DataType.FP32 || accumulatorType == DataType.FP64;
            boolean isIntegralAccumulator = accumulatorType.isIntegral();
            if (isFloatAccumulator) {
                return;
            }
            if (!isIntegralAccumulator || accumulatorType.byteSize() < inputType.byteSize()) {
                throw new IllegalArgumentException(
                        "Integral "
                                + reductionName
                                + " accumulator must be >= input type or FP32/FP64, got "
                                + accumulatorType
                                + " for input "
                                + inputType);
            }
            return;
        }
        throw new IllegalArgumentException(
                "Unsupported " + reductionName + " input type: " + inputType);
    }

    private DataType scalarType(Number scalar) {
        if (scalar instanceof Float || scalar instanceof Double) {
            return DataType.FP32;
        }
        if (scalar instanceof Byte || scalar instanceof Short || scalar instanceof Integer) {
            return DataType.I32;
        }
        if (scalar instanceof Long) {
            return DataType.I32;
        }
        return DataType.FP32;
    }

    private ExprNode maybeCast(ExprNode node, DataType targetType, Layout layout, Device device) {
        if (node.dataType() == targetType) {
            return node;
        }
        return new CastNode(node, targetType, layout, device);
    }

    private void requireSameType(TraceTensor left, TraceTensor right, String opName) {
        if (left.dataType() != right.dataType()) {
            throw new IllegalArgumentException(
                    opName
                            + " requires same data types, got "
                            + left.dataType()
                            + " and "
                            + right.dataType());
        }
    }

    private void requireBool(DataType dataType, String opName) {
        if (dataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires BOOL data type, got " + dataType);
        }
    }

    private void requireIntegral(DataType dataType, String opName) {
        if (!dataType.isIntegral() || dataType == DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires integral data type, got " + dataType);
        }
    }

    private void requirePrimitive(DataType dataType, String opName) {
        boolean primitive =
                dataType == DataType.BOOL
                        || dataType == DataType.I8
                        || dataType == DataType.I16
                        || dataType == DataType.I32
                        || dataType == DataType.I64
                        || dataType == DataType.FP16
                        || dataType == DataType.BF16
                        || dataType == DataType.FP32
                        || dataType == DataType.FP64;
        if (!primitive) {
            throw new IllegalArgumentException(
                    opName + " requires primitive data type, got " + dataType);
        }
    }

    private Layout requireCompatibleLayout(TraceTensor left, TraceTensor right) {
        if (!left.layout().isCongruentWith(right.layout())) {
            throw new IllegalArgumentException(
                    "Incompatible layouts: " + left.layout() + " vs " + right.layout());
        }
        return left.layout();
    }

    private Device requireCompatibleDevice(TraceTensor left, TraceTensor right) {
        if (!left.device().equals(right.device())) {
            throw new IllegalArgumentException(
                    "Incompatible devices: " + left.device() + " vs " + right.device());
        }
        return left.device();
    }

    private TraceTensor requireTrace(Tensor tensor) {
        if (tensor instanceof TraceTensor traceTensor) {
            return traceTensor;
        }
        Optional<LazyComputation> computation = tensor.computation();
        if (computation.isPresent() && computation.get() instanceof ConstantComputation constant) {
            ScalarNode node =
                    new ScalarNode(
                            constant.value(), tensor.dataType(), tensor.layout(), tensor.device());
            return new TraceTensor(node);
        }
        throw new IllegalArgumentException("Tracing requires trace tensors, got: " + tensor);
    }

    private UnsupportedOperationException unsupported(String name) {
        return new UnsupportedOperationException("Tracing does not support " + name + " yet");
    }
}

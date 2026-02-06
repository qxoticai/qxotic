package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.TypeRules;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.impl.ViewTransforms;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.CastOp;
import ai.qxotic.jota.ir.tir.Contiguous;
import ai.qxotic.jota.ir.tir.ReductionOperator;
import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import ai.qxotic.jota.ir.tir.ViewTransform;
import ai.qxotic.jota.memory.MemoryDomain;

final class IRTensorOps implements TensorOps {

    @Override
    public MemoryDomain<?> memoryDomain() {
        throw new UnsupportedOperationException("IR-T ops do not expose a memory domain");
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.ADD);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.SUBTRACT);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MULTIPLY);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.DIVIDE);
    }

    @Override
    public Tensor min(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MIN);
    }

    @Override
    public Tensor max(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MAX);
    }

    @Override
    public Tensor negate(Tensor x) {
        return unaryOp(x, UnaryOperator.NEGATE);
    }

    @Override
    public Tensor abs(Tensor x) {
        return unaryOp(x, UnaryOperator.ABS);
    }

    @Override
    public Tensor exp(Tensor x) {
        return unaryOp(x, UnaryOperator.EXP);
    }

    @Override
    public Tensor log(Tensor x) {
        return unaryOp(x, UnaryOperator.LOG);
    }

    @Override
    public Tensor sqrt(Tensor x) {
        IRTensor tensor = requireIRTensor(x);
        requireFloatingPoint(tensor.dataType(), "sqrt");
        return unaryOp(x, UnaryOperator.SQRT);
    }

    @Override
    public Tensor sin(Tensor x) {
        IRTensor tensor = requireIRTensor(x);
        requireFloatingPoint(tensor.dataType(), "sin");
        return unaryOp(x, UnaryOperator.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        IRTensor tensor = requireIRTensor(x);
        requireFloatingPoint(tensor.dataType(), "cos");
        return unaryOp(x, UnaryOperator.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        IRTensor tensor = requireIRTensor(x);
        requireFloatingPoint(tensor.dataType(), "tanh");
        return unaryOp(x, UnaryOperator.TANH);
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOperator.RECIPROCAL);
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        IRTensor tensor = requireIRTensor(x);
        requireIntegral(tensor.dataType(), "bitwiseNot");
        return unaryOp(x, UnaryOperator.BITWISE_NOT);
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOperator.BITWISE_AND, "bitwiseAnd");
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOperator.BITWISE_OR, "bitwiseOr");
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        return bitwiseBinaryOp(a, b, BinaryOperator.BITWISE_XOR, "bitwiseXor");
    }

    @Override
    public Tensor logicalNot(Tensor x) {
        return unaryOp(x, UnaryOperator.LOGICAL_NOT);
    }

    @Override
    public Tensor logicalAnd(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOperator.LOGICAL_AND, "logicalAnd");
    }

    @Override
    public Tensor logicalOr(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOperator.LOGICAL_OR, "logicalOr");
    }

    @Override
    public Tensor logicalXor(Tensor a, Tensor b) {
        return booleanBinaryOp(a, b, BinaryOperator.LOGICAL_XOR, "logicalXor");
    }

    @Override
    public Tensor equal(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.EQUAL);
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.LESS_THAN);
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        IRTensor condTensor = requireIRTensor(condition);
        IRTensor aTensor = requireIRTensor(trueValue);
        IRTensor bTensor = requireIRTensor(falseValue);

        // Validate condition is BOOL type
        requireBool(condTensor.dataType(), "where condition");

        // Validate true and false values have same type
        requireSameType(aTensor, bTensor, "where");

        Shape outputShape = resolveWhereShape(condTensor.shape(), aTensor.shape(), bTensor.shape());
        ai.qxotic.jota.ir.tir.TernaryOp node =
                new ai.qxotic.jota.ir.tir.TernaryOp(
                        ai.qxotic.jota.ir.tir.TernaryOperator.WHERE,
                        condTensor.node(),
                        aTensor.node(),
                        bTensor.node(),
                        outputShape);
        return new IRTensor(node, aTensor.device());
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(
                x, ReductionOperator.SUM, getAxes(_axis, _axes), keepDims, accumulatorType);
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(
                x, ReductionOperator.PROD, getAxes(_axis, _axes), keepDims, accumulatorType);
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        throw new UnsupportedOperationException(
                "mean not directly supported in IR-T. "
                        + "Use sum() / size() or sum().div(count).");
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.MAX, getAxes(_axis, _axes), keepDims, null);
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.MIN, getAxes(_axis, _axes), keepDims, null);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("matmul not supported in IR-T yet");
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("batchedMatmul not supported in IR-T yet");
    }

    @Override
    public Tensor viewTransform(Tensor input, ViewTransforms.ViewTransformSpec spec) {
        IRTensor tensor = requireIRTensor(input);
        TIRNode node =
                new ViewTransform(
                        tensor.node(), spec.kind(), spec.layout(), spec.needsLazyIndexing());
        return new IRTensor(node, tensor.device());
    }

    @Override
    public Tensor reshape(Tensor input, Shape newShape) {
        IRTensor tensor = requireIRTensor(input);
        Shape inputShape = tensor.shape();
        if (inputShape.size() != newShape.size()) {
            throw new IllegalArgumentException(
                    "Cannot reshape from "
                            + inputShape
                            + " to "
                            + newShape
                            + ": size mismatch ("
                            + inputShape.size()
                            + " vs "
                            + newShape.size()
                            + ")");
        }
        if (inputShape.isScalar()) {
            return broadcast(tensor, newShape);
        }
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(tensor.layout(), newShape);
        return viewTransform(tensor, spec);
    }

    @Override
    public Tensor to(Tensor input, Device targetDevice) {
        throw new UnsupportedOperationException(
                "Device transfer not supported in IR-T. "
                        + "Materialize tensor on source device then use Tensor.to() outside IR-T.");
    }

    @Override
    public Tensor contiguous(Tensor input) {
        IRTensor tensor = requireIRTensor(input);
        if (tensor.layout().isSuffixContiguous(0)) {
            return tensor;
        }
        TIRNode node = new Contiguous(tensor.node(), tensor.shape());
        return new IRTensor(node, tensor.device());
    }

    @Override
    public Tensor cast(Tensor input, DataType targetType) {
        IRTensor tensor = requireIRTensor(input);
        TIRNode node = new CastOp(tensor.node(), targetType, tensor.shape());
        return new IRTensor(node, tensor.device());
    }

    private Tensor unaryOp(Tensor x, UnaryOperator op) {
        IRTensor tensor = requireIRTensor(x);
        TIRNode node = new ai.qxotic.jota.ir.tir.UnaryOp(op, tensor.node(), tensor.shape());
        return new IRTensor(node, tensor.device());
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOperator op) {
        IRTensor left = requireIRTensor(a);
        IRTensor right = requireIRTensor(b);
        Shape outputShape = requireCompatibleShape(left, right);
        DataType targetType = TypeRules.promote(left.dataType(), right.dataType());
        TIRNode leftNode = maybeCast(left.node(), targetType);
        TIRNode rightNode = maybeCast(right.node(), targetType);
        TIRNode node = new ai.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensor(node, left.device());
    }

    private Tensor booleanBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensor left = requireIRTensor(a);
        IRTensor right = requireIRTensor(b);
        requireSameType(left, right, opName);
        Shape outputShape = requireCompatibleShape(left, right);
        TIRNode leftNode = left.node();
        TIRNode rightNode = right.node();
        TIRNode node = new ai.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensor(node, left.device());
    }

    private Tensor reduceAxes(
            Tensor input,
            ReductionOperator op,
            int[] _axes,
            boolean keepDims,
            DataType accumulatorType) {
        IRTensor tensor = requireIRTensor(input);
        int rank = tensor.shape().rank();
        // Wrap around axes
        int[] axes = new int[_axes.length];
        for (int i = 0; i < _axes.length; i++) {
            axes[i] = Util.wrapAround(_axes[i], rank);
        }
        // Determine accumulator type: use provided type or default to input type
        DataType accType = (accumulatorType != null) ? accumulatorType : tensor.dataType();
        Shape outputShape = reduceShape(tensor.shape(), axes, keepDims);
        TIRNode node =
                new ai.qxotic.jota.ir.tir.ReductionOp(
                        op, tensor.node(), axes, keepDims, accType, outputShape);
        return new IRTensor(node, tensor.device());
    }

    private int[] getAxes(int _axis, int... _axes) {
        int[] result = new int[_axes.length + 1];
        result[0] = _axis;
        System.arraycopy(_axes, 0, result, 1, _axes.length);
        return result;
    }

    private TIRNode maybeCast(TIRNode node, DataType targetType) {
        if (node.dataType() == targetType) {
            return node;
        }
        return new CastOp(node, targetType, node.shape());
    }

    private void requireSameType(IRTensor left, IRTensor right, String opName) {
        if (left.dataType() != right.dataType()) {
            throw new IllegalArgumentException(
                    opName
                            + " requires same data types, got: "
                            + left.dataType()
                            + " vs "
                            + right.dataType());
        }
    }

    private Shape requireCompatibleShape(IRTensor left, IRTensor right) {
        // Only true scalars (shape.isScalar()) can be broadcast, not arbitrary broadcasted tensors
        boolean leftIsTrueScalar = left.shape().isScalar();
        boolean rightIsTrueScalar = right.shape().isScalar();

        if (leftIsTrueScalar && rightIsTrueScalar) {
            return left.shape();
        }
        if (leftIsTrueScalar && !rightIsTrueScalar) {
            return right.shape();
        }
        if (!leftIsTrueScalar && rightIsTrueScalar) {
            return left.shape();
        }

        // Neither is a true scalar - shapes must be congruent
        if (!left.shape().isCongruentWith(right.shape())) {
            throw new IllegalArgumentException(
                    "Incompatible shapes: "
                            + left.shape()
                            + " vs "
                            + right.shape()
                            + ". Note: Only true scalar tensors (shape.isScalar() == true) can be broadcast, "
                            + "not broadcasted tensors. Use Tensor.scalar(value) to create scalar values.");
        }
        return left.shape();
    }

    private IRTensor requireIRTensor(Tensor tensor) {
        if (tensor instanceof IRTensor irtensor) {
            return irtensor;
        }
        throw new IllegalArgumentException("Expected IRTensor, got: " + tensor.getClass());
    }

    private void requireIntegral(DataType dataType, String opName) {
        if (!dataType.isIntegral() || dataType == DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires integral data type, got " + dataType);
        }
    }

    private void requireFloatingPoint(DataType dataType, String opName) {
        if (!dataType.isFloatingPoint()) {
            throw new IllegalArgumentException(
                    opName + " requires floating-point data type, got " + dataType);
        }
    }

    private void requireBool(DataType dataType, String opName) {
        if (dataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires BOOL data type, got " + dataType);
        }
    }

    private Tensor bitwiseBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensor left = requireIRTensor(a);
        IRTensor right = requireIRTensor(b);
        requireSameType(left, right, opName);
        requireIntegral(left.dataType(), opName);
        Shape outputShape = requireCompatibleShape(left, right);
        TIRNode leftNode = left.node();
        TIRNode rightNode = right.node();
        TIRNode node = new ai.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensor(node, left.device());
    }

    private Shape resolveWhereShape(Shape condShape, Shape trueShape, Shape falseShape) {
        Shape valueShape = requireCompatibleShape(trueShape, falseShape);
        if (condShape.isScalar()) {
            return valueShape;
        }
        if (valueShape.isScalar()) {
            return condShape;
        }
        if (!condShape.isCongruentWith(valueShape)) {
            throw new IllegalArgumentException(
                    "Incompatible shapes in where(): condition shape "
                            + condShape
                            + " is not compatible with value shapes "
                            + valueShape
                            + ". Only true scalar tensors can be broadcast.");
        }
        return valueShape;
    }

    private Shape requireCompatibleShape(Shape left, Shape right) {
        boolean leftIsTrueScalar = left.isScalar();
        boolean rightIsTrueScalar = right.isScalar();
        if (leftIsTrueScalar && rightIsTrueScalar) {
            return left;
        }
        if (leftIsTrueScalar && !rightIsTrueScalar) {
            return right;
        }
        if (!leftIsTrueScalar && rightIsTrueScalar) {
            return left;
        }
        if (!left.isCongruentWith(right)) {
            throw new IllegalArgumentException(
                    "Incompatible shapes: "
                            + left
                            + " vs "
                            + right
                            + ". Note: Only true scalar tensors (shape.isScalar() == true) can be broadcast, "
                            + "not broadcasted tensors. Use Tensor.scalar(value) to create scalar values.");
        }
        return left;
    }

    private Shape reduceShape(Shape inputShape, int[] axes, boolean keepDims) {
        int inputRank = inputShape.rank();
        boolean[] reduced = new boolean[inputRank];
        for (int axis : axes) {
            if (axis >= 0 && axis < inputRank) {
                reduced[axis] = true;
            }
        }
        if (keepDims) {
            long[] dims = new long[inputRank];
            for (int i = 0; i < inputRank; i++) {
                dims[i] = reduced[i] ? 1 : inputShape.flatAt(i);
            }
            return Shape.flat(dims);
        }
        long[] dims = new long[inputRank - axes.length];
        int idx = 0;
        for (int i = 0; i < inputRank; i++) {
            if (!reduced[i]) {
                dims[idx++] = inputShape.flatAt(i);
            }
        }
        return Shape.flat(dims);
    }
}

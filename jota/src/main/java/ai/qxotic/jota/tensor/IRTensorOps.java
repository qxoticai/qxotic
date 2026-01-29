package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.TypeRules;
import ai.qxotic.jota.ir.irt.BinaryOperator;
import ai.qxotic.jota.ir.irt.CastOp;
import ai.qxotic.jota.ir.irt.Contiguous;
import ai.qxotic.jota.ir.irt.IRTNode;
import ai.qxotic.jota.ir.irt.ReductionOperator;
import ai.qxotic.jota.ir.irt.UnaryOperator;
import ai.qxotic.jota.ir.irt.ViewTransform;
import ai.qxotic.jota.memory.MemoryContext;

final class IRTensorOps implements TensorOps {

    @Override
    public MemoryContext<?> context() {
        throw new UnsupportedOperationException("IR-T ops do not expose a memory context");
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.ADD, "add");
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.SUBTRACT, "subtract");
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MULTIPLY, "multiply");
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.DIVIDE, "divide");
    }

    @Override
    public Tensor min(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MIN, "min");
    }

    @Override
    public Tensor max(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.MAX, "max");
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
        return unaryOp(x, UnaryOperator.SQRT);
    }

    @Override
    public Tensor sin(Tensor x) {
        return unaryOp(x, UnaryOperator.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        return unaryOp(x, UnaryOperator.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        return unaryOp(x, UnaryOperator.TANH);
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOperator.RECIPROCAL);
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        return unaryOp(x, UnaryOperator.BITWISE_NOT);
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.BITWISE_AND, "bitwiseAnd");
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.BITWISE_OR, "bitwiseOr");
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.BITWISE_XOR, "bitwiseXor");
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
        return binaryOp(a, b, BinaryOperator.EQUAL, "equal");
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOperator.LESS_THAN, "lessThan");
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        IRTensor condTensor = requireIRTensor(condition);
        IRTensor aTensor = requireIRTensor(trueValue);
        IRTensor bTensor = requireIRTensor(falseValue);
        Layout layout = requireCompatibleLayout(aTensor, bTensor);
        ai.qxotic.jota.ir.irt.TernaryOp node =
                new ai.qxotic.jota.ir.irt.TernaryOp(
                        ai.qxotic.jota.ir.irt.TernaryOperator.WHERE,
                        condTensor.node(),
                        aTensor.node(),
                        bTensor.node());
        return new IRTensor(node, aTensor.device());
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.SUM, getAxes(_axis, _axes), keepDims, accumulatorType);
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.PROD, getAxes(_axis, _axes), keepDims, accumulatorType);
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        throw new UnsupportedOperationException(
                "mean not directly supported in IR-T. "
                        + "Use sum() / size() or sum().div(count).");
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.MAX, getAxes(_axis, _axes), keepDims, null);
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOp.MIN, getAxes(_axis, _axes), keepDims, null);
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
    public Tensor viewTransform(Tensor input, Layout layout, long byteOffsetDelta, String hint) {
        IRTensor tensor = requireIRTensor(input);
        IRTNode node = new ViewTransform(tensor.node(), hint, layout);
        return new IRTensor(node, tensor.device());
    }

    @Override
    public Tensor reshape(Tensor input, Shape newShape) {
        IRTensor tensor = requireIRTensor(input);
        Shape inputShape = tensor.layout().shape();
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
        Layout newLayout = Layout.of(newShape, ai.qxotic.jota.Stride.rowMajor(newShape));
        IRTNode node = new ViewTransform(tensor.node(), "view", newLayout);
        return new IRTensor(node, tensor.device());
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
        IRTNode node = new Contiguous(tensor.node());
        return new IRTensor(node, tensor.device());
    }

    @Override
    public Tensor cast(Tensor input, DataType targetType) {
        IRTensor tensor = requireIRTensor(input);
        IRTNode node = new CastOp(tensor.node(), targetType);
        return new IRTensor(node, tensor.device());
    }

    private Tensor unaryOp(Tensor x, UnaryOperator op) {
        IRTensor tensor = requireIRTensor(x);
        IRTNode node = new ai.qxotic.jota.ir.irt.UnaryOp(op, tensor.node());
        return new IRTensor(node, tensor.device());
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensor left = requireIRTensor(a);
        IRTensor right = requireIRTensor(b);
        Layout layout = requireCompatibleLayout(left, right);
        DataType targetType = TypeRules.promote(left.dataType(), right.dataType());
        IRTNode leftNode = maybeCast(left.node(), targetType);
        IRTNode rightNode = maybeCast(right.node(), targetType);
        IRTNode node = new ai.qxotic.jota.ir.irt.BinaryOp(op, leftNode, rightNode);
        return new IRTensor(node, left.device());
    }

    private Tensor booleanBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensor left = requireIRTensor(a);
        IRTensor right = requireIRTensor(b);
        requireSameType(left, right, opName);
        Layout layout = requireCompatibleLayout(left, right);
        IRTNode leftNode = left.node();
        IRTNode rightNode = right.node();
        IRTNode node = new ai.qxotic.jota.ir.irt.BinaryOp(op, leftNode, rightNode);
        return new IRTensor(node, left.device());
    }

    private Tensor reduceAxes(
            Tensor input, ReductionOp op, int[] axes, boolean keepDims, DataType accumulatorType) {
        IRTensor tensor = requireIRTensor(input);
        ReductionOperator irtOp = mapReductionOp(op);
        IRTNode node = new ai.qxotic.jota.ir.irt.ReductionOp(irtOp, tensor.node(), axes, keepDims);
        return new IRTensor(node, tensor.device());
    }

    private int[] getAxes(int _axis, int... _axes) {
        if (_axes.length == 0 && _axis < 0) {
            return new int[0];
        }
        int[] result = new int[_axes.length + 1];
        result[0] = _axis;
        System.arraycopy(_axes, 0, result, 1, _axes.length);
        return result;
    }

    private IRTNode maybeCast(IRTNode node, DataType targetType) {
        if (node.dataType() == targetType) {
            return node;
        }
        return new CastOp(node, targetType);
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

    private Layout requireCompatibleLayout(IRTensor left, IRTensor right) {
        boolean leftIsScalar = left.layout().shape().isScalar();
        boolean rightIsScalar = right.layout().shape().isScalar();

        if (leftIsScalar && rightIsScalar) {
            return left.layout();
        }
        if (leftIsScalar && !rightIsScalar) {
            return right.layout();
        }
        if (!leftIsScalar && rightIsScalar) {
            return left.layout();
        }

        if (!left.layout().isCongruentWith(right.layout())) {
            throw new IllegalArgumentException(
                    "Incompatible layouts: "
                            + left.layout().shape()
                            + " vs "
                            + right.layout().shape());
        }
        return left.layout();
    }

    private IRTensor requireIRTensor(Tensor tensor) {
        if (tensor instanceof IRTensor irtensor) {
            return irtensor;
        }
        throw new IllegalArgumentException("Expected IRTensor, got: " + tensor.getClass());
    }

    private ReductionOperator mapReductionOp(ReductionOp op) {
        if (op == ReductionOp.SUM) {
            return ReductionOperator.SUM;
        } else if (op == ReductionOp.PROD) {
            return ReductionOperator.PROD;
        } else if (op == ReductionOp.MIN) {
            return ReductionOperator.MIN;
        } else if (op == ReductionOp.MAX) {
            return ReductionOperator.MAX;
        } else {
            throw new IllegalArgumentException("Unknown reduction op: " + op);
        }
    }
}

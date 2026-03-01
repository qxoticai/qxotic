package com.qxotic.jota.tensor;

import com.qxotic.jota.*;
import com.qxotic.jota.impl.ViewTransforms;
import com.qxotic.jota.ir.tir.*;

final class IRTensorOps implements TensorOps {

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
        IRTensorImpl tensor = requireIRTensor(x);
        TensorTypeSemantics.requireFloatingPoint(tensor.dataType(), "sqrt");
        return unaryOp(x, UnaryOperator.SQRT);
    }

    @Override
    public Tensor sin(Tensor x) {
        IRTensorImpl tensor = requireIRTensor(x);
        TensorTypeSemantics.requireFloatingPoint(tensor.dataType(), "sin");
        return unaryOp(x, UnaryOperator.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        IRTensorImpl tensor = requireIRTensor(x);
        TensorTypeSemantics.requireFloatingPoint(tensor.dataType(), "cos");
        return unaryOp(x, UnaryOperator.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        IRTensorImpl tensor = requireIRTensor(x);
        TensorTypeSemantics.requireFloatingPoint(tensor.dataType(), "tanh");
        return unaryOp(x, UnaryOperator.TANH);
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOperator.RECIPROCAL);
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        IRTensorImpl tensor = requireIRTensor(x);
        TensorTypeSemantics.requireIntegral(tensor.dataType(), "bitwiseNot");
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
    public Tensor leftShift(Tensor a, Tensor b) {
        return shiftBinaryOp(a, b, BinaryOperator.SHIFT_LEFT, "leftShift");
    }

    @Override
    public Tensor rightShift(Tensor a, Tensor b) {
        return shiftBinaryOp(a, b, BinaryOperator.SHIFT_RIGHT, "rightShift");
    }

    @Override
    public Tensor rightShiftUnsigned(Tensor a, Tensor b) {
        return shiftBinaryOp(a, b, BinaryOperator.SHIFT_RIGHT_UNSIGNED, "rightShiftUnsigned");
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
        return compareBinaryOp(a, b, BinaryOperator.EQUAL);
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return compareBinaryOp(a, b, BinaryOperator.LESS_THAN);
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        IRTensorImpl condTensor = requireIRTensor(condition);
        IRTensorImpl aTensor = requireIRTensor(trueValue);
        IRTensorImpl bTensor = requireIRTensor(falseValue);

        // Validate condition is BOOL type
        TensorTypeSemantics.requireBool(condTensor.dataType(), "where condition");

        // Validate true and false values have same type
        requireSameType(aTensor, bTensor, "where");

        Shape outputShape =
                TensorSemantics.resolveWhereShape(
                        condTensor.shape(), aTensor.shape(), bTensor.shape());
        TernaryOp node =
                new TernaryOp(
                        TernaryOperator.WHERE,
                        condTensor.node(),
                        aTensor.node(),
                        bTensor.node(),
                        outputShape);
        return new IRTensorImpl(node, aTensor.device());
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.SUM, _axis, _axes, keepDims, accumulatorType);
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.PROD, _axis, _axes, keepDims, accumulatorType);
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.MAX, _axis, _axes, keepDims, null);
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return reduceAxes(x, ReductionOperator.MIN, _axis, _axes, keepDims, null);
    }

    @Override
    public Tensor gather(Tensor input, Tensor indices, int _axis) {
        IRTensorImpl inputTensor = requireIRTensor(input);
        IRTensorImpl indicesTensor = requireIRTensor(indices);

        // Validate indices is integral type
        DataType indicesType = indicesTensor.dataType();
        if (!indicesType.isIntegral()) {
            throw new IllegalArgumentException(
                    "Gather indices must be integral type, got " + indicesType);
        }

        // Compute output shape
        Shape outputShape =
                GatherOp.computeOutputShape(inputTensor.shape(), indicesTensor.shape(), _axis);

        // Create GatherOp node
        GatherOp node = new GatherOp(inputTensor.node(), indicesTensor.node(), _axis, outputShape);

        return new IRTensorImpl(node, inputTensor.device());
    }

    @Override
    public Tensor dot(Tensor a, Tensor b, DataType accumulatorType) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);

        TensorTypeSemantics.requireNumericNonBool(left.dataType(), "dot");
        TensorTypeSemantics.requireNumericNonBool(right.dataType(), "dot");
        requireSameType(left, right, "dot");

        if (left.shape().rank() != 1 || right.shape().rank() != 1) {
            throw new IllegalArgumentException(
                    "dot requires rank-1 tensors, got " + left.shape() + " and " + right.shape());
        }
        long length = left.shape().flatAt(0);
        long rightLength = right.shape().flatAt(0);
        if (length != rightLength) {
            throw new IllegalArgumentException(
                    "dot requires equal vector lengths, got " + length + " and " + rightLength);
        }
        if (length == 0) {
            throw new IllegalArgumentException("dot requires non-empty vectors");
        }

        DataType accType =
                TensorTypeSemantics.resolveReductionAccumulator(
                        left.dataType(), accumulatorType, "dot");
        Tensor castLeft = cast(left, accType);
        Tensor castRight = cast(right, accType);
        Tensor product = multiply(castLeft, castRight);
        return sum(product, accType);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        if (left.shape().rank() != 2 || right.shape().rank() != 2) {
            throw new IllegalArgumentException(
                    "matmul requires rank-2 tensors, got "
                            + left.shape()
                            + " and "
                            + right.shape());
        }

        long m = left.shape().flatAt(0);
        long k = left.shape().flatAt(1);
        long rightK = right.shape().flatAt(0);
        long n = right.shape().flatAt(1);
        if (k != rightK) {
            throw new IllegalArgumentException(
                    "matmul inner dimensions must match, got " + k + " and " + rightK);
        }

        DataType dtype = left.dataType();
        if (dtype != right.dataType()) {
            throw new IllegalArgumentException(
                    "matmul requires matching input dtypes, got "
                            + dtype
                            + " and "
                            + right.dataType());
        }

        Tensor leftPrepared = contiguous(left);
        Tensor rightPrepared = contiguous(right);
        IRTensorImpl leftTensor = requireIRTensor(leftPrepared);
        IRTensorImpl rightTensor = requireIRTensor(rightPrepared);

        Shape productShape = Shape.of(m, k, n);
        Tensor leftExpanded = view(leftTensor, Shape.of(m, k, 1));
        Tensor rightExpanded = view(rightTensor, Shape.of(1, k, n));
        Tensor leftBroadcast = broadcast(leftExpanded, productShape);
        Tensor rightBroadcast = broadcast(rightExpanded, productShape);
        Tensor product = multiply(leftBroadcast, rightBroadcast);
        return sum(product, dtype, 1);
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        if (left.shape().rank() != 3 || right.shape().rank() != 3) {
            throw new IllegalArgumentException(
                    "batchedMatmul requires rank-3 tensors, got "
                            + left.shape()
                            + " and "
                            + right.shape());
        }

        long batch = left.shape().flatAt(0);
        long rightBatch = right.shape().flatAt(0);
        long m = left.shape().flatAt(1);
        long k = left.shape().flatAt(2);
        long rightK = right.shape().flatAt(1);
        long n = right.shape().flatAt(2);
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

        DataType dtype = left.dataType();
        if (dtype != right.dataType()) {
            throw new IllegalArgumentException(
                    "batchedMatmul requires matching input dtypes, got "
                            + dtype
                            + " and "
                            + right.dataType());
        }

        Tensor leftPrepared = contiguous(left);
        Tensor rightPrepared = contiguous(right);
        IRTensorImpl leftTensor = requireIRTensor(leftPrepared);
        IRTensorImpl rightTensor = requireIRTensor(rightPrepared);

        Shape productShape = Shape.of(batch, m, k, n);
        Tensor leftExpanded = view(leftTensor, Shape.of(batch, m, k, 1));
        Tensor rightExpanded = view(rightTensor, Shape.of(batch, 1, k, n));
        Tensor leftBroadcast = broadcast(leftExpanded, productShape);
        Tensor rightBroadcast = broadcast(rightExpanded, productShape);
        Tensor product = multiply(leftBroadcast, rightBroadcast);
        return sum(product, dtype, 2);
    }

    @Override
    public Tensor viewTransform(Tensor input, ViewTransforms.ViewTransformSpec spec) {
        IRTensorImpl tensor = requireIRTensor(input);
        TIRNode node =
                new ViewTransform(
                        tensor.node(), spec.kind(), spec.layout(), spec.needsLazyIndexing());
        return new IRTensorImpl(node, tensor.device());
    }

    @Override
    public Tensor reshape(Tensor input, Shape newShape) {
        IRTensorImpl tensor = requireIRTensor(input);
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
        IRTensorImpl tensor = requireIRTensor(input);
        if (tensor.layout().isSuffixContiguous(0)) {
            return tensor;
        }
        TIRNode node = new Contiguous(tensor.node(), tensor.shape());
        return new IRTensorImpl(node, tensor.device());
    }

    @Override
    public Tensor cast(Tensor input, DataType targetType) {
        IRTensorImpl tensor = requireIRTensor(input);
        TIRNode node = new CastOp(tensor.node(), targetType, tensor.shape());
        return new IRTensorImpl(node, tensor.device());
    }

    private Tensor unaryOp(Tensor x, UnaryOperator op) {
        IRTensorImpl tensor = requireIRTensor(x);
        TIRNode node = new com.qxotic.jota.ir.tir.UnaryOp(op, tensor.node(), tensor.shape());
        return new IRTensorImpl(node, tensor.device());
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOperator op) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        Shape outputShape = requireCompatibleShape(left, right);
        DataType targetType =
                TensorTypeSemantics.promoteForArithmetic(
                        left.dataType(), right.dataType(), op.name());
        TIRNode leftNode = maybeCast(left.node(), targetType);
        TIRNode rightNode = maybeCast(right.node(), targetType);
        TIRNode node = new com.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensorImpl(node, left.device());
    }

    private Tensor compareBinaryOp(Tensor a, Tensor b, BinaryOperator op) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        Shape outputShape = requireCompatibleShape(left, right);
        DataType targetType =
                TensorTypeSemantics.promoteForComparison(
                        left.dataType(), right.dataType(), op.name());
        TIRNode leftNode = maybeCast(left.node(), targetType);
        TIRNode rightNode = maybeCast(right.node(), targetType);
        TIRNode node = new com.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensorImpl(node, left.device());
    }

    private Tensor booleanBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        TensorTypeSemantics.requireBooleanPair(left.dataType(), right.dataType(), opName);
        Shape outputShape = requireCompatibleShape(left, right);
        TIRNode leftNode = left.node();
        TIRNode rightNode = right.node();
        TIRNode node = new com.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensorImpl(node, left.device());
    }

    private Tensor reduceAxes(
            Tensor input,
            ReductionOperator op,
            int firstAxis,
            int[] otherAxes,
            boolean keepDims,
            DataType accumulatorType) {
        IRTensorImpl tensor = requireIRTensor(input);
        int[] axes =
                TensorSemantics.normalizeReductionAxes(tensor.shape().rank(), firstAxis, otherAxes);
        DataType accType =
                TensorTypeSemantics.resolveReductionAccumulator(
                        tensor.dataType(), accumulatorType, op.name().toLowerCase());
        Shape outputShape = TensorSemantics.reduceShape(tensor.shape(), axes, keepDims);
        TIRNode node = new ReductionOp(op, tensor.node(), axes, keepDims, accType, outputShape);
        return new IRTensorImpl(node, tensor.device());
    }

    private TIRNode maybeCast(TIRNode node, DataType targetType) {
        if (node.dataType() == targetType) {
            return node;
        }
        return new CastOp(node, targetType, node.shape());
    }

    private void requireSameType(IRTensorImpl left, IRTensorImpl right, String opName) {
        if (left.dataType() != right.dataType()) {
            throw new IllegalArgumentException(
                    opName
                            + " requires same data types, got: "
                            + left.dataType()
                            + " vs "
                            + right.dataType());
        }
    }

    private Shape requireCompatibleShape(IRTensorImpl left, IRTensorImpl right) {
        return TensorSemantics.requireCompatibleShape(left.shape(), right.shape());
    }

    private IRTensorImpl requireIRTensor(Tensor tensor) {
        if (tensor instanceof IRTensorImpl irtensor) {
            return irtensor;
        }
        throw new IllegalArgumentException("Expected IRTensorImpl, got: " + tensor.getClass());
    }

    private Tensor bitwiseBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        TensorTypeSemantics.requireSameIntegralType(left.dataType(), right.dataType(), opName);
        Shape outputShape = requireCompatibleShape(left, right);
        TIRNode leftNode = left.node();
        TIRNode rightNode = right.node();
        TIRNode node = new com.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensorImpl(node, left.device());
    }

    private Tensor shiftBinaryOp(Tensor a, Tensor b, BinaryOperator op, String opName) {
        IRTensorImpl left = requireIRTensor(a);
        IRTensorImpl right = requireIRTensor(b);
        TensorTypeSemantics.requireShiftOperandTypes(left.dataType(), right.dataType(), opName);
        Shape outputShape = requireCompatibleShape(left, right);
        TIRNode leftNode = left.node();
        TIRNode rightNode = maybeCast(right.node(), DataType.I32);
        TIRNode node = new com.qxotic.jota.ir.tir.BinaryOp(op, leftNode, rightNode, outputShape);
        return new IRTensorImpl(node, left.device());
    }
}

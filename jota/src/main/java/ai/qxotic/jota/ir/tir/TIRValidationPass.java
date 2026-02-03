package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;

/**
 * Validation pass for IR-T graphs.
 *
 * <p>Checks shape compatibility, reduction axes, and view consistency. This pass does not modify
 * the graph.
 */
public final class TIRValidationPass implements TIRPass {

    @Override
    public TIRGraph run(TIRGraph graph) {
        Objects.requireNonNull(graph, "graph");
        Set<TIRNode> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        for (TIRNode output : graph.outputs()) {
            validateNode(output, visited);
        }
        return graph;
    }

    private void validateNode(TIRNode node, Set<TIRNode> visited) {
        if (!visited.add(node)) {
            return;
        }
        switch (node) {
            case TensorInput __ -> {
                // Input is validated by construction.
            }
            case ScalarInput __ -> {
                // Scalar inputs are validated by construction.
            }
            case ScalarConstant sc -> validateScalarConstant(sc);
            case IotaConstant ic -> validateIotaConstant(ic);
            case UnaryOp op -> {
                validateUnaryOp(op);
                validateNode(op.input(), visited);
            }
            case BinaryOp op -> {
                validateBinaryOp(op);
                validateNode(op.left(), visited);
                validateNode(op.right(), visited);
            }
            case TernaryOp op -> {
                validateTernaryOp(op);
                validateNode(op.cond(), visited);
                validateNode(op.trueExpr(), visited);
                validateNode(op.falseExpr(), visited);
            }
            case CastOp op -> {
                validateCastOp(op);
                validateNode(op.input(), visited);
            }
            case ReductionOp op -> {
                validateReductionOp(op);
                validateNode(op.input(), visited);
            }
            case ViewTransform view -> {
                validateViewTransform(view);
                validateNode(view.input(), visited);
            }
            case Contiguous contig -> {
                validateContiguous(contig);
                validateNode(contig.input(), visited);
            }
        }
    }

    private void validateScalarConstant(ScalarConstant node) {
        Objects.requireNonNull(node.shape(), "ScalarConstant shape cannot be null");
    }

    private void validateIotaConstant(IotaConstant node) {
        if (node.shape().size() != node.count()) {
            throw new IllegalArgumentException(
                    "IotaConstant count mismatch: count="
                            + node.count()
                            + " shape.size="
                            + node.shape().size());
        }
    }

    private void validateUnaryOp(UnaryOp node) {
        requireShapeEquals("UnaryOp", node.shape(), node.input().shape());
    }

    private void validateBinaryOp(BinaryOp node) {
        Shape expected = BinaryOp.broadcastShapes(node.left().shape(), node.right().shape());
        requireShapeEquals("BinaryOp", node.shape(), expected);
    }

    private void validateTernaryOp(TernaryOp node) {
        if (node.op() == TernaryOperator.WHERE && node.cond().dataType() != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "WHERE condition must be BOOL, got " + node.cond().dataType());
        }
        Shape valueShape =
                BinaryOp.broadcastShapes(node.trueExpr().shape(), node.falseExpr().shape());
        Shape expected = BinaryOp.broadcastShapes(node.cond().shape(), valueShape);
        requireShapeEquals("TernaryOp", node.shape(), expected);
    }

    private void validateCastOp(CastOp node) {
        requireShapeEquals("CastOp", node.shape(), node.input().shape());
    }

    private void validateContiguous(Contiguous node) {
        requireShapeEquals("Contiguous", node.shape(), node.input().shape());
    }

    private void validateReductionOp(ReductionOp node) {
        int[] axes = node.axes();
        if (axes.length == 0) {
            throw new IllegalArgumentException("Reduction axes cannot be empty");
        }
        Shape inputShape = node.input().shape();
        int rank = inputShape.rank();
        boolean[] seen = new boolean[rank];
        for (int axis : axes) {
            if (axis < 0 || axis >= rank) {
                throw new IllegalArgumentException(
                        "Reduction axis out of bounds: " + axis + " for rank " + rank);
            }
            if (seen[axis]) {
                throw new IllegalArgumentException(
                        "Reduction axes must be unique, duplicate: " + axis);
            }
            seen[axis] = true;
        }

        Shape expected = reduceShape(inputShape, axes, node.keepDims());
        requireShapeEquals("ReductionOp", node.shape(), expected);
    }

    private void validateViewTransform(ViewTransform node) {
        Shape inputShape = node.input().shape();
        Shape outputShape = node.shape();
        switch (node.kind()) {
            case ViewKind.Reshape reshape -> {
                requireShapeEquals("ViewTransform(reshape input)", inputShape, reshape.fromShape());
                requireShapeEquals("ViewTransform(reshape output)", outputShape, reshape.toShape());
            }
            case ViewKind.Broadcast broadcast -> {
                requireShapeEquals(
                        "ViewTransform(broadcast input)", inputShape, broadcast.fromShape());
                requireShapeEquals(
                        "ViewTransform(broadcast output)", outputShape, broadcast.toShape());
            }
            case ViewKind.Expand expand -> {
                requireShapeEquals("ViewTransform(expand input)", inputShape, expand.fromShape());
                requireShapeEquals("ViewTransform(expand output)", outputShape, expand.toShape());
            }
            case ViewKind.Transpose transpose -> {
                int[] perm = transpose.permutation();
                if (perm.length != inputShape.rank()) {
                    throw new IllegalArgumentException(
                            "Transpose permutation length "
                                    + perm.length
                                    + " does not match rank "
                                    + inputShape.rank());
                }
                Shape expected = inputShape.permute(perm);
                requireShapeEquals("ViewTransform(transpose output)", outputShape, expected);
            }
            case ViewKind.Slice slice -> {
                int axis = slice.axis();
                if (axis < 0 || axis >= inputShape.rank()) {
                    throw new IllegalArgumentException(
                            "Slice axis out of bounds: " + axis + " for rank " + inputShape.rank());
                }
                long inDim = inputShape.flatAt(axis);
                long outDim = outputShape.flatAt(axis);
                if (outDim > inDim) {
                    throw new IllegalArgumentException(
                            "Slice output dim "
                                    + outDim
                                    + " exceeds input dim "
                                    + inDim
                                    + " on axis "
                                    + axis);
                }
            }
        }
    }

    private Shape reduceShape(Shape inputShape, int[] axes, boolean keepDims) {
        int rank = inputShape.rank();
        boolean[] reduced = new boolean[rank];
        for (int axis : axes) {
            if (axis >= 0 && axis < rank) {
                reduced[axis] = true;
            }
        }
        if (keepDims) {
            long[] dims = new long[rank];
            for (int i = 0; i < rank; i++) {
                dims[i] = reduced[i] ? 1 : inputShape.flatAt(i);
            }
            return Shape.flat(dims);
        }
        long[] dims = new long[rank - axes.length];
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            if (!reduced[i]) {
                dims[idx++] = inputShape.flatAt(i);
            }
        }
        return Shape.flat(dims);
    }

    private void requireShapeEquals(String label, Shape actual, Shape expected) {
        if (!actual.equals(expected)) {
            throw new IllegalArgumentException(
                    label + " shape mismatch: expected " + expected + " but got " + actual);
        }
    }
}

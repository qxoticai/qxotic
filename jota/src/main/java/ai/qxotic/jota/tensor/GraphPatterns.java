package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;

public final class GraphPatterns {

    private GraphPatterns() {}

    public static BinaryOpInfo requireBinaryOp(ExpressionGraph graph) {
        if (!(graph.root() instanceof BinaryNode root)) {
            throw new UnsupportedOperationException("Expected binary root");
        }
        if (!(root.left() instanceof InputNode left)
                || !(root.right() instanceof InputNode right)) {
            throw new UnsupportedOperationException("Expected input nodes for binary op");
        }
        return new BinaryOpInfo(
                root.op(), left.index(), right.index(), left.dataType(), right.dataType());
    }

    public static UnaryOpInfo requireUnaryOp(ExpressionGraph graph) {
        if (!(graph.root() instanceof UnaryNode root)) {
            throw new UnsupportedOperationException("Expected unary root");
        }
        if (!(root.input() instanceof InputNode input)) {
            throw new UnsupportedOperationException("Expected input node for unary op");
        }
        return new UnaryOpInfo(root.op(), input.index());
    }

    public record BinaryOpInfo(
            BinaryOp op, int leftIndex, int rightIndex, DataType leftType, DataType rightType) {}

    public record UnaryOpInfo(UnaryOp op, int inputIndex) {}
}

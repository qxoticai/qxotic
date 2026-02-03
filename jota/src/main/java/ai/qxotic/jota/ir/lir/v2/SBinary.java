package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;

public final class SBinary extends ScalarNode {
    private final BinaryOperator op;

    SBinary(int id, BinaryOperator op, V2Node left, V2Node right, DataType dataType) {
        super(id, V2Kind.S_BINARY, dataType, new V2Node[] {left, right}, true, graphCommutative(op));
        this.op = op;
    }

    public BinaryOperator op() {
        return op;
    }

    public V2Node left() {
        return inputs()[0];
    }

    public V2Node right() {
        return inputs()[1];
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        V2Node left = left();
        V2Node right = right();

        if (isCommutative() && left.id() > right.id()) {
            return graph.scalarBinary(op, right, left);
        }

        if (left == right && (op == BinaryOperator.MIN || op == BinaryOperator.MAX)) {
            return left;
        }

        if (left instanceof SConst leftConst && right instanceof SConst rightConst) {
            return graph.foldBinary(op, leftConst, rightConst);
        }

        return graph.simplifyBinary(op, left, right);
    }

    private static boolean graphCommutative(BinaryOperator op) {
        return switch (op) {
            case ADD,
                    MULTIPLY,
                    MIN,
                    MAX,
                    LOGICAL_AND,
                    LOGICAL_OR,
                    LOGICAL_XOR,
                    BITWISE_AND,
                    BITWISE_OR,
                    BITWISE_XOR,
                    EQUAL -> true;
            default -> false;
        };
    }
}

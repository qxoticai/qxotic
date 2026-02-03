package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.ir.lir.IndexBinary.IndexBinaryOp;

public final class IBinary extends IndexNode {
    private final IndexBinaryOp op;

    IBinary(int id, IndexBinaryOp op, V2Node left, V2Node right) {
        super(id, V2Kind.I_BINARY, new V2Node[] {left, right}, true, isCommutative(op));
        this.op = op;
    }

    public IndexBinaryOp op() {
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
            return graph.indexBinary(op, right, left);
        }

        if (left instanceof IConst lc && right instanceof IConst rc) {
            return graph.foldIndexBinary(op, lc, rc);
        }

        return graph.simplifyIndexBinary(op, left, right);
    }

    private static boolean isCommutative(IndexBinaryOp op) {
        return switch (op) {
            case ADD, MULTIPLY -> true;
            default -> false;
        };
    }
}

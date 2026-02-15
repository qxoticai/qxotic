package ai.qxotic.jota.ir.lir;

public final class IBinary extends IndexNode {
    private final IndexBinaryOp op;

    IBinary(int id, IndexBinaryOp op, LIRExprNode left, LIRExprNode right) {
        super(id, LIRExprKind.I_BINARY, new LIRExprNode[] {left, right}, true, isCommutative(op));
        this.op = op;
    }

    public IndexBinaryOp op() {
        return op;
    }

    public LIRExprNode left() {
        return inputs()[0];
    }

    public LIRExprNode right() {
        return inputs()[1];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        LIRExprNode left = left();
        LIRExprNode right = right();

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

package ai.qxotic.jota.ir.lir;

public final class IFromScalar extends IndexNode {
    IFromScalar(int id, LIRExprNode scalarExpr) {
        super(id, LIRExprKind.I_FROM_SCALAR, new LIRExprNode[] {scalarExpr}, true, false);
    }

    public LIRExprNode scalarExpr() {
        return inputs()[0];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

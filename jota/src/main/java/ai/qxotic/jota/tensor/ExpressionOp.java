package ai.qxotic.jota.tensor;

final class ExpressionOp implements Op {

    static final ExpressionOp INSTANCE = new ExpressionOp();

    private ExpressionOp() {}

    @Override
    public String name() {
        return "expression";
    }

    @Override
    public OpKind kind() {
        return OpKind.SPECIAL;
    }
}

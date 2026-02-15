package ai.qxotic.jota.ir.lir;

public final class IVar extends IndexNode {
    private final String name;

    IVar(int id, String name) {
        super(id, LIRExprKind.I_VAR, new LIRExprNode[0], true, false);
        this.name = name;
    }

    public String name() {
        return name;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

package ai.qxotic.jota.ir.lir.v2;

public final class IVar extends IndexNode {
    private final String name;

    IVar(int id, String name) {
        super(id, V2Kind.I_VAR, new V2Node[0], true, false);
        this.name = name;
    }

    public String name() {
        return name;
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}

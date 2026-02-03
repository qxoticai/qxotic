package ai.qxotic.jota.ir.lir.v2;

public final class IConst extends IndexNode {
    private final long value;

    IConst(int id, long value) {
        super(id, V2Kind.I_CONST, new V2Node[0], true, false);
        this.value = value;
    }

    public long value() {
        return value;
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}

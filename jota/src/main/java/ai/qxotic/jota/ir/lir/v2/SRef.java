package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class SRef extends ScalarNode {
    private final String name;

    SRef(int id, String name, DataType dataType) {
        super(id, V2Kind.S_REF, dataType, new V2Node[0], true, false);
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

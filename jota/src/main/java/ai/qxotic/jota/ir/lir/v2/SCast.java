package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class SCast extends ScalarNode {
    private final DataType targetType;

    SCast(int id, V2Node input, DataType targetType) {
        super(id, V2Kind.S_CAST, targetType, new V2Node[] {input}, true, false);
        this.targetType = targetType;
    }

    public V2Node input() {
        return inputs()[0];
    }

    public DataType targetType() {
        return targetType;
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        V2Node input = input();
        if (input instanceof SConst constant) {
            return graph.foldCast(constant, targetType);
        }
        if (input.dataType() == targetType) {
            return input;
        }
        return this;
    }
}

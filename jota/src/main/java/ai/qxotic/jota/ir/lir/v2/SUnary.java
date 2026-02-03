package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.UnaryOperator;

public final class SUnary extends ScalarNode {
    private final UnaryOperator op;

    SUnary(int id, UnaryOperator op, V2Node input, DataType dataType) {
        super(id, V2Kind.S_UNARY, dataType, new V2Node[] {input}, true, false);
        this.op = op;
    }

    public UnaryOperator op() {
        return op;
    }

    public V2Node input() {
        return inputs()[0];
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        V2Node input = input();
        if (input instanceof SConst constant) {
            return graph.foldUnary(op, constant);
        }
        return this;
    }
}

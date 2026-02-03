package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class STernary extends ScalarNode {
    STernary(int id, V2Node condition, V2Node trueValue, V2Node falseValue, DataType dataType) {
        super(id, V2Kind.S_TERNARY, dataType, new V2Node[] {condition, trueValue, falseValue}, true, false);
    }

    public V2Node condition() {
        return inputs()[0];
    }

    public V2Node trueValue() {
        return inputs()[1];
    }

    public V2Node falseValue() {
        return inputs()[2];
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        V2Node condition = condition();
        V2Node trueValue = trueValue();
        V2Node falseValue = falseValue();

        if (trueValue == falseValue) {
            return trueValue;
        }

        if (condition instanceof SConst constant) {
            return graph.foldTernary(constant, trueValue, falseValue);
        }

        return this;
    }
}

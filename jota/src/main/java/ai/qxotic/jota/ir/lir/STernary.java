package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public final class STernary extends ScalarNode {
    STernary(
            int id,
            LIRExprNode condition,
            LIRExprNode trueValue,
            LIRExprNode falseValue,
            DataType dataType) {
        super(
                id,
                LIRExprKind.S_TERNARY,
                dataType,
                new LIRExprNode[] {condition, trueValue, falseValue},
                true,
                false);
    }

    public LIRExprNode condition() {
        return inputs()[0];
    }

    public LIRExprNode trueValue() {
        return inputs()[1];
    }

    public LIRExprNode falseValue() {
        return inputs()[2];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        LIRExprNode condition = condition();
        LIRExprNode trueValue = trueValue();
        LIRExprNode falseValue = falseValue();

        if (trueValue == falseValue) {
            return trueValue;
        }

        if (condition instanceof SConst constant) {
            return graph.foldTernary(constant, trueValue, falseValue);
        }

        return this;
    }
}

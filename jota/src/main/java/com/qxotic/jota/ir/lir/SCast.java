package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;

public final class SCast extends ScalarNode {
    private final DataType targetType;

    SCast(int id, LIRExprNode input, DataType targetType) {
        super(id, LIRExprKind.S_CAST, targetType, new LIRExprNode[] {input}, true, false);
        this.targetType = targetType;
    }

    public LIRExprNode input() {
        return inputs()[0];
    }

    public DataType targetType() {
        return targetType;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        LIRExprNode input = input();
        if (input instanceof SConst constant) {
            return graph.foldCast(constant, targetType);
        }
        if (input.dataType() == targetType) {
            return input;
        }
        return this;
    }
}

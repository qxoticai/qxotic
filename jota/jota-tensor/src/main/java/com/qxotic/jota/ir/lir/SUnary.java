package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.tir.UnaryOperator;

public final class SUnary extends ScalarNode {
    private final UnaryOperator op;

    SUnary(int id, UnaryOperator op, LIRExprNode input, DataType dataType) {
        super(id, LIRExprKind.S_UNARY, dataType, new LIRExprNode[] {input}, true, false);
        this.op = op;
    }

    public UnaryOperator op() {
        return op;
    }

    public LIRExprNode input() {
        return inputs()[0];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        LIRExprNode input = input();
        if (input instanceof SConst constant) {
            return graph.foldUnary(op, constant);
        }
        return this;
    }
}

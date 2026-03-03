package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;

public final class SFromIndex extends ScalarNode {
    SFromIndex(int id, LIRExprNode indexExpr, DataType dataType) {
        super(id, LIRExprKind.S_FROM_INDEX, dataType, new LIRExprNode[] {indexExpr}, true, false);
    }

    public LIRExprNode indexExpr() {
        return inputs()[0];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

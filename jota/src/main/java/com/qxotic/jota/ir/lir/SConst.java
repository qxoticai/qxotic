package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;

public final class SConst extends ScalarNode {
    private final long rawBits;

    SConst(int id, long rawBits, DataType dataType) {
        super(id, LIRExprKind.S_CONST, dataType, new LIRExprNode[0], true, false);
        this.rawBits = rawBits;
    }

    public long rawBits() {
        return rawBits;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

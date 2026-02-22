package com.qxotic.jota.ir.lir;

final class Use {
    final LIRExprNode user;
    final int inputIndex;
    Use next;

    Use(LIRExprNode user, int inputIndex, Use next) {
        this.user = user;
        this.inputIndex = inputIndex;
        this.next = next;
    }
}

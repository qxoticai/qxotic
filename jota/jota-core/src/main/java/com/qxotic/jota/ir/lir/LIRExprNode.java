package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;

public abstract class LIRExprNode {
    private final int id;
    private final LIRExprKind kind;
    private final DataType dataType;
    private final LIRExprNode[] inputs;
    private final boolean pure;
    private final boolean commutative;
    private Use uses;
    private int useCount;
    private LIRExprNode replacement;

    LIRExprNode(
            int id,
            LIRExprKind kind,
            DataType dataType,
            LIRExprNode[] inputs,
            boolean pure,
            boolean commutative) {
        this.id = id;
        this.kind = kind;
        this.dataType = dataType;
        this.inputs = inputs;
        this.pure = pure;
        this.commutative = commutative;
    }

    public final int id() {
        return id;
    }

    public final LIRExprKind kind() {
        return kind;
    }

    public final DataType dataType() {
        return dataType;
    }

    public final LIRExprNode[] inputs() {
        return inputs;
    }

    public final boolean isPure() {
        return pure;
    }

    public final boolean isCommutative() {
        return commutative;
    }

    public final Use uses() {
        return uses;
    }

    public final int useCount() {
        return useCount;
    }

    public final void addUse(LIRExprNode user, int inputIndex) {
        uses = new Use(user, inputIndex, uses);
        useCount++;
    }

    public final void clearUses() {
        uses = null;
        useCount = 0;
    }

    public final void replaceInput(int inputIndex, LIRExprNode newInput) {
        inputs[inputIndex] = newInput;
    }

    public final LIRExprNode replacement() {
        return replacement;
    }

    public final void setReplacement(LIRExprNode replacement) {
        this.replacement = replacement;
    }

    public abstract LIRExprNode canonicalize(LIRExprGraph graph);
}

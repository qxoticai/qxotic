package com.qxotic.jota.ir.lir;

import java.util.List;
import java.util.Objects;

/** A sequence of statements executed in order. */
public final class Block extends LIRExprNode {
    private final List<LIRExprNode> statements;

    Block(int id, List<LIRExprNode> statements) {
        super(
                id,
                LIRExprKind.BLOCK,
                null,
                Objects.requireNonNull(statements, "statements cannot be null")
                        .toArray(new LIRExprNode[0]),
                false,
                false);
        this.statements = List.copyOf(statements);
    }

    /** Creates a block from varargs statements. */
    public static Block of(int id, LIRExprNode... statements) {
        return new Block(id, List.of(statements));
    }

    public List<LIRExprNode> statements() {
        return statements;
    }

    /** Returns true if this block is empty. */
    public boolean isEmpty() {
        return statements.isEmpty();
    }

    /** Returns the number of statements. */
    public int size() {
        return statements.size();
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

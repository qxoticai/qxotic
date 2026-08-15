package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.Block;
import com.qxotic.jota.ir.lir.IConst;
import com.qxotic.jota.ir.lir.LIRExprGraph;
import com.qxotic.jota.ir.lir.LIRExprKind;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.Yield;

public final class CLikeSourceSupport {
    private static final int MAX_INLINE_EXPR_LEN = 80;

    private CLikeSourceSupport() {}

    public static boolean isConstZero(LIRExprNode node, LIRExprGraph exprGraph) {
        LIRExprNode resolved = exprGraph.resolve(node);
        return resolved instanceof IConst ic && ic.value() == 0;
    }

    public static boolean isConstOne(LIRExprNode node, LIRExprGraph exprGraph) {
        LIRExprNode resolved = exprGraph.resolve(node);
        return resolved instanceof IConst ic && ic.value() == 1;
    }

    public static Yield extractYield(Block body) {
        if (!body.statements().isEmpty()) {
            LIRExprNode last = body.statements().getLast();
            if (last instanceof Yield yield) {
                return yield;
            }
        }
        throw new IllegalStateException("Structured loop body must end with Yield");
    }

    public static boolean shouldMaterializeTemp(LIRExprNode node, String expr) {
        if (node.kind() == LIRExprKind.S_TERNARY) {
            return true;
        }
        if (node.useCount() > 2) {
            return true;
        }
        if (expr.length() <= MAX_INLINE_EXPR_LEN) {
            return false;
        }
        return switch (node.kind()) {
            case S_UNARY, S_BINARY, S_CAST -> true;
            default -> false;
        };
    }
}

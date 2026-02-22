package com.qxotic.jota.ir.tir;

import java.util.List;

final class TIRNodeUtils {

    private TIRNodeUtils() {}

    static List<TIRNode> inputsOf(TIRNode node) {
        return switch (node) {
            case UnaryOp op -> List.of(op.input());
            case BinaryOp op -> List.of(op.left(), op.right());
            case TernaryOp op -> List.of(op.cond(), op.trueExpr(), op.falseExpr());
            case CastOp op -> List.of(op.input());
            case ReductionOp op -> List.of(op.input());
            case GatherOp op -> List.of(op.input(), op.indices());
            case ViewTransform vt -> List.of(vt.input());
            case Contiguous contig -> List.of(contig.input());
            default -> List.of();
        };
    }

    static boolean isComputeNode(TIRNode node) {
        return switch (node) {
            case UnaryOp __ -> true;
            case BinaryOp __ -> true;
            case TernaryOp __ -> true;
            case CastOp __ -> true;
            case ReductionOp __ -> true;
            case GatherOp __ -> true;
            case Contiguous __ -> true;
            default -> false;
        };
    }

    static boolean isGraphInputNode(TIRNode node) {
        return node instanceof TensorInput || node instanceof ScalarInput;
    }
}

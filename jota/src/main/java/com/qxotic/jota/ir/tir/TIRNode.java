package com.qxotic.jota.ir.tir;

import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.base.IRNode;

/**
 * Base interface for all IR-T (Tensor IR) nodes. IR-T represents tensor operations with explicit
 * shapes and data types. Layout is only carried by input and view nodes.
 */
public sealed interface TIRNode extends IRNode
        permits TensorInput,
                ScalarInput,
                ScalarConstant,
                UnaryOp,
                BinaryOp,
                TernaryOp,
                CastOp,
                ReductionOp,
                GatherOp,
                ViewTransform,
                Contiguous,
                IotaConstant {

    /** Returns the shape of this node. */
    Shape shape();

    /** Accepts a visitor for traversing the IR-T graph. */
    default <T> T accept(TIRVisitor<T> visitor) {
        return switch (this) {
            case TensorInput n -> visitor.visitTensorInput(n);
            case ScalarInput n -> visitor.visitScalarInput(n);
            case ScalarConstant n -> visitor.visitScalarConstant(n);
            case UnaryOp n -> visitor.visitUnaryOp(n);
            case BinaryOp n -> visitor.visitBinaryOp(n);
            case TernaryOp n -> visitor.visitTernaryOp(n);
            case CastOp n -> visitor.visitCastOp(n);
            case ReductionOp n -> visitor.visitReductionOp(n);
            case GatherOp n -> visitor.visitGatherOp(n);
            case ViewTransform n -> visitor.visitViewTransform(n);
            case Contiguous n -> visitor.visitContiguous(n);
            case IotaConstant n -> visitor.visitIotaConstant(n);
        };
    }
}

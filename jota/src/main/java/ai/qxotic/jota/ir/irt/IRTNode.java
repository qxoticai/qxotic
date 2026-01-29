package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.base.IRNode;

/**
 * Base interface for all IR-T (Tensor IR) nodes. IR-T represents purely semantic tensor operations
 * with shape and layout information.
 */
public sealed interface IRTNode extends IRNode
        permits TensorInput,
                ScalarConstant,
                UnaryOp,
                BinaryOp,
                TernaryOp,
                CastOp,
                ReductionOp,
                ViewTransform,
                Contiguous,
                IotaConstant {

    /** Returns the layout of this node (shape + stride). */
    Layout layout();

    /** Accepts a visitor for traversing the IR-T graph. */
    default <T> T accept(IRTVisitor<T> visitor) {
        return switch (this) {
            case TensorInput n -> visitor.visitTensorInput(n);
            case ScalarConstant n -> visitor.visitScalarConstant(n);
            case UnaryOp n -> visitor.visitUnaryOp(n);
            case BinaryOp n -> visitor.visitBinaryOp(n);
            case TernaryOp n -> visitor.visitTernaryOp(n);
            case CastOp n -> visitor.visitCastOp(n);
            case ReductionOp n -> visitor.visitReductionOp(n);
            case ViewTransform n -> visitor.visitViewTransform(n);
            case Contiguous n -> visitor.visitContiguous(n);
            case IotaConstant n -> visitor.visitIotaConstant(n);
        };
    }
}

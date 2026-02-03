package ai.qxotic.jota.ir.tir;

/** Visitor interface for traversing IR-T graphs. */
public interface TIRVisitor<T> {

    T visitTensorInput(TensorInput node);

    T visitScalarInput(ScalarInput node);

    T visitUnaryOp(UnaryOp node);

    T visitBinaryOp(BinaryOp node);

    T visitTernaryOp(TernaryOp node);

    T visitCastOp(CastOp node);

    T visitReductionOp(ReductionOp node);

    T visitViewTransform(ViewTransform node);

    T visitContiguous(Contiguous node);

    T visitScalarConstant(ScalarConstant node);

    T visitIotaConstant(IotaConstant node);
}

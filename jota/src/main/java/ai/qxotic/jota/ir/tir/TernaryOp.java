package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Ternary operation node in IR-T (e.g., where/select). */
public record TernaryOp(
        TernaryOperator op, TIRNode cond, TIRNode trueExpr, TIRNode falseExpr, Shape shape)
        implements TIRNode {

    public TernaryOp(TernaryOperator op, TIRNode cond, TIRNode trueExpr, TIRNode falseExpr) {
        this(op, cond, trueExpr, falseExpr, broadcastShapes(cond, trueExpr, falseExpr));
    }

    public TernaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(cond);
        Objects.requireNonNull(trueExpr);
        Objects.requireNonNull(falseExpr);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return trueExpr.dataType();
    }

    @Override
    public Shape shape() {
        return shape;
    }

    private static Shape broadcastShapes(TIRNode cond, TIRNode trueExpr, TIRNode falseExpr) {
        Shape valueShape = BinaryOp.broadcastShapes(trueExpr.shape(), falseExpr.shape());
        return BinaryOp.broadcastShapes(cond.shape(), valueShape);
    }
}

package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Ternary operation node in IR-T (e.g., where/select). */
public record TernaryOp(TernaryOperator op, TIRNode cond, TIRNode trueExpr, TIRNode falseExpr)
        implements TIRNode {

    public TernaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(cond);
        Objects.requireNonNull(trueExpr);
        Objects.requireNonNull(falseExpr);
    }

    @Override
    public DataType dataType() {
        return trueExpr.dataType();
    }

    @Override
    public Layout layout() {
        // Output must be row-major, not a broadcast layout with zero strides.
        Shape shape = trueExpr.layout().shape();
        return Layout.rowMajor(shape);
    }
}

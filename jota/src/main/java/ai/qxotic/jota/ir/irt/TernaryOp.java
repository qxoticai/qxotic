package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Objects;

/** Ternary operation node in IR-T (e.g., where/select). */
public record TernaryOp(TernaryOperator op, IRTNode cond, IRTNode trueExpr, IRTNode falseExpr)
        implements IRTNode {

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
        return trueExpr.layout();
    }
}

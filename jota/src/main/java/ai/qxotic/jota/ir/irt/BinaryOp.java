package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Objects;

/** Binary operation node in IR-T. */
public record BinaryOp(BinaryOperator op, IRTNode left, IRTNode right) implements IRTNode {

    public BinaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(left);
        Objects.requireNonNull(right);
    }

    @Override
    public DataType dataType() {
        return left.dataType();
    }

    @Override
    public Layout layout() {
        return left.layout();
    }
}

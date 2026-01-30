package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
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
        // Output must be row-major, not a broadcast layout with zero strides.
        // Use left's shape but ensure proper row-major strides.
        Shape shape = left.layout().shape();
        return Layout.rowMajor(shape);
    }
}

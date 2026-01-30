package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Binary operation node in IR-T. */
public record BinaryOp(BinaryOperator op, TIRNode left, TIRNode right) implements TIRNode {

    public BinaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(left);
        Objects.requireNonNull(right);
    }

    @Override
    public DataType dataType() {
        return switch (op) {
            case EQUAL, LESS_THAN -> ai.qxotic.jota.DataType.BOOL;
            default -> left.dataType();
        };
    }

    @Override
    public Layout layout() {
        // Output must be row-major, not a broadcast layout with zero strides.
        // Use left's shape but ensure proper row-major strides.
        Shape shape = left.layout().shape();
        return Layout.rowMajor(shape);
    }
}

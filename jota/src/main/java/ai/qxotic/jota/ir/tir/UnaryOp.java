package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Unary operation node in IR-T. */
public record UnaryOp(UnaryOperator op, TIRNode input) implements TIRNode {

    public UnaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(input);
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Layout layout() {
        // Output must be row-major, not a broadcast layout with zero strides.
        Shape shape = input.layout().shape();
        return Layout.rowMajor(shape);
    }
}

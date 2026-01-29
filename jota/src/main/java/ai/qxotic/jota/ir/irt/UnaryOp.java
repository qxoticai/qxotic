package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Objects;

/** Unary operation node in IR-T. */
public record UnaryOp(UnaryOperator op, IRTNode input) implements IRTNode {

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
        return input.layout();
    }
}

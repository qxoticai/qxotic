package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Objects;

/** Cast operation node in IR-T. */
public record CastOp(IRTNode input, DataType targetDtype) implements IRTNode {

    public CastOp {
        Objects.requireNonNull(input);
        Objects.requireNonNull(targetDtype);
    }

    @Override
    public DataType dataType() {
        return targetDtype;
    }

    @Override
    public Layout layout() {
        return input.layout();
    }
}

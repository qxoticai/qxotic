package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Objects;

/** Cast operation node in IR-T. */
public record CastOp(TIRNode input, DataType targetDataType) implements TIRNode {

    public CastOp {
        Objects.requireNonNull(input);
        Objects.requireNonNull(targetDataType);
    }

    @Override
    public DataType dataType() {
        return targetDataType;
    }

    @Override
    public Layout layout() {
        // Compute operations produce row-major outputs.
        return Layout.rowMajor(input.layout().shape());
    }
}

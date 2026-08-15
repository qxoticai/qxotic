package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;
import java.util.Objects;

/** Loop-carried argument for structured loops. */
public record LoopIterArg(String name, DataType dataType, LIRExprNode init) {

    public LoopIterArg {
        Objects.requireNonNull(name, "name cannot be null");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(init, "init cannot be null");
    }
}

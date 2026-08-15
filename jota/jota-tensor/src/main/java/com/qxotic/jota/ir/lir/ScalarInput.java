package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;
import java.util.Objects;

/** Represents a scalar input parameter passed by value at runtime. */
public record ScalarInput(int id, DataType dataType) implements LIRInput {

    public ScalarInput {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        if (id < 0) {
            throw new IllegalArgumentException("id must be non-negative, got: " + id);
        }
    }
}

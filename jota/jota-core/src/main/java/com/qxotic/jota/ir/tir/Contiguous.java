package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Objects;

/**
 * Contiguous operation in IR-T. Represents a semantic requirement that the tensor should have
 * contiguous row-major layout. IR-L will decide whether to emit a no-op (if already contiguous) or
 * allocate+copy.
 */
public record Contiguous(TIRNode input, Shape shape) implements TIRNode {

    public Contiguous(TIRNode input) {
        this(input, input.shape());
    }

    public Contiguous {
        Objects.requireNonNull(input);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Shape shape() {
        return shape;
    }
}

package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.FftType;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.fft}. Generated from the op table; see {@link Op}. */
public record FftOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.fft";

    public FftOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code fft_type}. */
    public FftType fftType() {
        return FftType.from(OpSupport.required(operation, "fft_type"));
    }

    /** Attribute {@code fft_length}. */
    public long[] fftLength() {
        return OpSupport.i64s(operation, "fft_length");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitFft(this);
    }
}

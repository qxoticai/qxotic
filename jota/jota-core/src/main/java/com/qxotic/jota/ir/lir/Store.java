package com.qxotic.jota.ir.lir;

import java.util.Objects;

/** Store a scalar value to a buffer at the given byte offset. */
public final class Store extends LIRExprNode {
    private final BufferRef buffer;

    Store(int id, BufferRef buffer, LIRExprNode offset, LIRExprNode value) {
        super(
                id,
                LIRExprKind.STORE,
                null,
                new LIRExprNode[] {
                    Objects.requireNonNull(offset, "offset cannot be null"),
                    Objects.requireNonNull(value, "value cannot be null")
                },
                false,
                false);
        this.buffer = Objects.requireNonNull(buffer, "buffer cannot be null");
    }

    public BufferRef buffer() {
        return buffer;
    }

    public LIRExprNode offset() {
        return inputs()[0];
    }

    public LIRExprNode value() {
        return inputs()[1];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

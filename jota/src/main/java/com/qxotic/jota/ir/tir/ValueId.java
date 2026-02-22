package com.qxotic.jota.ir.tir;

/** Stable identifier for a materialized scheduled value. */
public record ValueId(int id) {

    public ValueId {
        if (id < 0) {
            throw new IllegalArgumentException("ValueId must be non-negative, got: " + id);
        }
    }
}

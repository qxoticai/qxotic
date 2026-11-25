package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.type.Type;
import java.util.Objects;

/**
 * An SSA value: an operation result or a block argument. Values have identity semantics; the
 * optional {@link #nameHint()} is the name used in the source text ({@code %arg0}, {@code %cst}),
 * printers may renumber.
 */
public final class Value {

    private final Type type;
    private final String nameHint;
    private Object owner; // Operation or Block
    private int index;

    Value(Type type, String nameHint) {
        this.type = Objects.requireNonNull(type, "type");
        this.nameHint = nameHint;
    }

    public Type type() {
        return type;
    }

    public String nameHint() {
        return nameHint;
    }

    void setOwner(Object owner, int index) {
        this.owner = owner;
        this.index = index;
    }

    /** The defining operation, or null for a block argument. */
    public Operation definingOp() {
        return owner instanceof Operation op ? op : null;
    }

    /** The owning block when this value is a block argument, otherwise null. */
    public Block ownerBlock() {
        return owner instanceof Block b ? b : null;
    }

    public boolean isBlockArgument() {
        return owner instanceof Block;
    }

    /** Result number or argument number within its owner. */
    public int index() {
        return index;
    }

    @Override
    public String toString() {
        return "%" + (nameHint != null ? nameHint : "?") + " : " + type.mlir();
    }
}

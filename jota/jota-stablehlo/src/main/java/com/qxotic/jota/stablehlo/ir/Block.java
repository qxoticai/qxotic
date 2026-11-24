package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** A block: typed arguments followed by a list of operations (the last one is a terminator). */
public final class Block {

    private final List<Value> arguments = new ArrayList<>();
    private final List<Operation> operations = new ArrayList<>();
    private String label;

    public Block() {}

    /** Appends a block argument and returns it. */
    public Value addArgument(Type type, String nameHint) {
        Value v = new Value(type, nameHint);
        v.setOwner(this, arguments.size());
        arguments.add(v);
        return v;
    }

    public void addOperation(Operation op) {
        operations.add(op);
    }

    public List<Value> arguments() {
        return Collections.unmodifiableList(arguments);
    }

    public Value argument(int i) {
        return arguments.get(i);
    }

    public List<Operation> operations() {
        return Collections.unmodifiableList(operations);
    }

    public Operation terminator() {
        return operations.isEmpty() ? null : operations.getLast();
    }

    /** Block label from the source ({@code bb0}), informational. */
    public String label() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
}

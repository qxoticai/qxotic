package ai.qxotic.jota.ir.lir;

import java.util.List;
import java.util.Objects;

/** A sequence of statements executed in order. */
public record Block(List<LIRNode> statements) implements LIRNode {

    public Block {
        Objects.requireNonNull(statements, "statements cannot be null");
        statements = List.copyOf(statements);
    }

    /** Creates a block from varargs statements. */
    public static Block of(LIRNode... statements) {
        return new Block(List.of(statements));
    }

    /** Returns true if this block is empty. */
    public boolean isEmpty() {
        return statements.isEmpty();
    }

    /** Returns the number of statements. */
    public int size() {
        return statements.size();
    }
}

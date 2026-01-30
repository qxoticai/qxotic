package ai.qxotic.jota.ir.irl;

import java.util.List;
import java.util.Objects;

/** A sequence of statements executed in order. */
public record Block(List<IRLNode> statements) implements IRLNode {

    public Block {
        Objects.requireNonNull(statements, "statements cannot be null");
        statements = List.copyOf(statements);
    }

    /** Creates a block from varargs statements. */
    public static Block of(IRLNode... statements) {
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

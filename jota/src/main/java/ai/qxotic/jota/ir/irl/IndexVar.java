package ai.qxotic.jota.ir.irl;

import java.util.Objects;

/** Loop index variable reference. */
public record IndexVar(String name) implements IndexExpr {

    public IndexVar {
        Objects.requireNonNull(name, "name cannot be null");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
    }
}

package ai.qxotic.jota.ir.lir;

import java.util.List;
import java.util.Objects;

/** Yield values from a structured loop body. Must be the last instruction in the loop body. */
public record Yield(List<ScalarExpr> values) implements LIRNode {

    public Yield {
        Objects.requireNonNull(values, "values cannot be null");
        values = List.copyOf(values);
        for (ScalarExpr value : values) {
            Objects.requireNonNull(value, "yield value cannot be null");
        }
    }

    /** Creates an empty yield. */
    public static Yield empty() {
        return new Yield(List.of());
    }
}

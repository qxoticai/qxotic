package ai.qxotic.jota.ir.lir;

import java.util.List;
import java.util.Objects;

/** Yield values from a structured loop body. Must be the last instruction in the loop body. */
public final class Yield extends LIRExprNode {
    private final List<LIRExprNode> values;

    Yield(int id, List<LIRExprNode> values) {
        super(
                id,
                LIRExprKind.YIELD,
                null,
                Objects.requireNonNull(values, "values cannot be null")
                        .toArray(new LIRExprNode[0]),
                false,
                false);
        for (LIRExprNode value : values) {
            Objects.requireNonNull(value, "yield value cannot be null");
        }
        this.values = List.copyOf(values);
    }

    public List<LIRExprNode> values() {
        return values;
    }

    /** Creates an empty yield. */
    public static Yield empty(int id) {
        return new Yield(id, List.of());
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}

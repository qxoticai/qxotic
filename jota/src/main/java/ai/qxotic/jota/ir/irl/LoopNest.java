package ai.qxotic.jota.ir.irl;

import java.util.List;
import java.util.Objects;

/**
 * A nest of loops that share iteration. This enables fusion analysis - loops in a nest can
 * potentially be fused if they have compatible bounds.
 */
public record LoopNest(List<Loop> loops, IRLNode body) implements IRLNode {

    public LoopNest {
        Objects.requireNonNull(loops, "loops cannot be null");
        if (loops.isEmpty()) {
            throw new IllegalArgumentException("loops cannot be empty");
        }
        loops = List.copyOf(loops);
        Objects.requireNonNull(body, "body cannot be null");
    }

    /** Creates a loop nest from varargs loops. */
    public static LoopNest of(IRLNode body, Loop... loops) {
        return new LoopNest(List.of(loops), body);
    }

    /** Returns the outermost loop. */
    public Loop outermost() {
        return loops.getFirst();
    }

    /** Returns the innermost loop. */
    public Loop innermost() {
        return loops.getLast();
    }

    /** Returns the nesting depth. */
    public int depth() {
        return loops.size();
    }
}

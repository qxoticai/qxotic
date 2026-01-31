package ai.qxotic.jota.ir.lir;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Manages a pipeline of LIR transformation passes. Passes are executed in the order they are added.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * LIRGraph optimized = new LIRPassManager()
 *     .add(new IndexSimplificationPass())
 *     .add(new CommonSubexpressionElimination())
 *     .run(graph);
 * }</pre>
 */
public final class LIRPassManager {

    private final List<LIRPass> passes = new ArrayList<>();

    /**
     * Adds a pass to the pipeline.
     *
     * @param pass the pass to add
     * @return this manager for chaining
     */
    public LIRPassManager add(LIRPass pass) {
        Objects.requireNonNull(pass, "pass cannot be null");
        passes.add(pass);
        return this;
    }

    /**
     * Runs all passes in sequence on the given graph.
     *
     * @param graph the input graph
     * @return the transformed graph
     */
    public LIRGraph run(LIRGraph graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        LIRGraph current = graph;
        for (LIRPass pass : passes) {
            current = pass.run(current);
        }
        return current;
    }

    /**
     * Returns the number of passes in the pipeline.
     *
     * @return the pass count
     */
    public int size() {
        return passes.size();
    }

    /**
     * Returns true if the pipeline is empty.
     *
     * @return true if no passes have been added
     */
    public boolean isEmpty() {
        return passes.isEmpty();
    }
}

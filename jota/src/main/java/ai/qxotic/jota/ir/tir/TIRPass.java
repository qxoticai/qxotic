package ai.qxotic.jota.ir.tir;

/**
 * Interface for TIR graph transformation passes.
 *
 * <p>TIR passes transform a TIRGraph into another TIRGraph, potentially simplifying or optimizing
 * the graph structure.
 */
@FunctionalInterface
public interface TIRPass {

    /**
     * Runs this pass on the given graph.
     *
     * @param graph the input graph
     * @return the transformed graph (may be the same instance if no changes were made)
     */
    TIRGraph run(TIRGraph graph);

    /**
     * Returns the name of this pass (for debugging/logging).
     *
     * @return the pass name
     */
    default String name() {
        return getClass().getSimpleName();
    }
}

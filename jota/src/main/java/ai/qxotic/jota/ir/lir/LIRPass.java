package ai.qxotic.jota.ir.lir;

/**
 * Interface for LIR graph transformation passes. Passes transform an LIRGraph into a new LIRGraph,
 * potentially optimizing or simplifying it.
 */
@FunctionalInterface
public interface LIRPass {

    /**
     * Transforms the given LIR graph.
     *
     * @param graph the input graph
     * @return the transformed graph (may be the same instance if no changes)
     */
    LIRGraph run(LIRGraph graph);

    /**
     * Returns a descriptive name for this pass.
     *
     * @return the pass name
     */
    default String name() {
        return getClass().getSimpleName();
    }
}

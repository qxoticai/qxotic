package ai.qxotic.jota.ir.lir;

/** Worklist pass that canonicalizes the unified expression graph. */
public final class LIRWorklistPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        graph.exprGraph().processWorklist();
        return graph;
    }

    @Override
    public String name() {
        return "LIRWorklistPass";
    }
}

package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.ir.TIRToLIRLowerer;
import java.util.ArrayList;
import java.util.List;

/**
 * Standard optimization pipeline for LIR graphs.
 *
 * <p>This class provides a pre-configured pipeline of optimization passes that should be applied
 * when lowering TIR to LIR. The passes are ordered to maximize optimization effectiveness:
 *
 * <ol>
 *   <li><b>LIRWorklistPass</b> - Unique-node DAG canonicalization and constant folding
 *   <li><b>LIRWorklistPass</b> - Unique-node DAG canonicalization and constant folding
 * </ol>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * // Simple usage with default pipeline
 * LIRGraph optimized = LIRStandardPipeline.optimize(lirGraph);
 *
 * // Custom usage with additional passes or iterations
 * LIRStandardPipeline pipeline = new LIRStandardPipeline()
 *     .withIterations(2)  // Run core passes twice
 *     .withPass(new MyCustomPass());  // Add custom pass
 * LIRGraph optimized = pipeline.run(lirGraph);
 * }</pre>
 *
 * @see TIRToLIRLowerer
 */
public class LIRStandardPipeline {

    private final List<LIRPass> passes;
    private int iterations = 1;
    private boolean verbose = false;
    private final boolean timing = false; // Boolean.getBoolean("jota.lir.timing");

    /** Creates a standard pipeline with default pass ordering. */
    public LIRStandardPipeline() {
        this.passes = createStandardPasses();
    }

    /**
     * Creates the standard list of passes in the correct order.
     *
     * @return list of passes
     */
    private List<LIRPass> createStandardPasses() {
        List<LIRPass> standardPasses = new ArrayList<>();

        // Core optimization passes (expression DAG based)
        standardPasses.add(new LIRCanonicalizerPass());
        standardPasses.add(new LIRCSEPass());
        return standardPasses;
    }

    /**
     * Sets the number of iterations to run the core optimization passes.
     *
     * <p>Multiple iterations can help when passes expose new optimization opportunities for each
     * other. The default is 1 iteration.
     *
     * @param iterations number of iterations (must be >= 1)
     * @return this pipeline for chaining
     */
    public LIRStandardPipeline withIterations(int iterations) {
        if (iterations < 1) {
            throw new IllegalArgumentException("iterations must be >= 1");
        }
        this.iterations = iterations;
        return this;
    }

    /**
     * Enables or disables verbose output showing each pass and its effect.
     *
     * @param verbose true to enable verbose output
     * @return this pipeline for chaining
     */
    public LIRStandardPipeline withVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * Adds a custom pass to the pipeline before dead code elimination.
     *
     * @param pass the custom pass to add
     * @return this pipeline for chaining
     */
    public LIRStandardPipeline withPass(LIRPass pass) {
        // Insert before the last pass (DeadCodeElimination)
        passes.add(passes.size() - 1, pass);
        return this;
    }

    /**
     * Adds a custom pass at a specific position in the pipeline.
     *
     * @param index the position to insert the pass
     * @param pass the custom pass to add
     * @return this pipeline for chaining
     */
    public LIRStandardPipeline withPassAt(int index, LIRPass pass) {
        passes.add(index, pass);
        return this;
    }

    /**
     * Runs the optimization pipeline on the given LIR graph.
     *
     * @param graph the input LIR graph
     * @return the optimized LIR graph
     */
    public LIRGraph run(LIRGraph graph) {
        LIRGraph result = graph;

        for (int i = 0; i < iterations; i++) {
            if (verbose && iterations > 1) {
                System.out.println("=== Iteration " + (i + 1) + " of " + iterations + " ===");
            }

            for (LIRPass pass : passes) {
                LIRGraph previous = result;
                long start = timing ? System.nanoTime() : 0L;
                result = pass.run(result);
                long elapsed = timing ? System.nanoTime() - start : 0L;

                if (timing) {
                    System.out.println(
                            "LIR pass "
                                    + pass.name()
                                    + " took "
                                    + (elapsed / 1_000_000.0)
                                    + " ms");
                }

                if (verbose) {
                    System.out.println("After " + pass.name() + ":");
                    System.out.println(new LIRTextRenderer().render(result));
                    System.out.println();
                }

                // Early termination if no changes
                if (result == previous && !verbose) {
                    // Graph unchanged, but we continue to run remaining passes
                    // as they might make changes even if this one didn't
                }
            }
        }

        return result;
    }

    /**
     * Convenience method to run the standard pipeline with default settings.
     *
     * <p>This is the recommended way to optimize a freshly lowered LIR graph.
     *
     * @param graph the input LIR graph
     * @return the optimized LIR graph
     */
    public static LIRGraph optimize(LIRGraph graph) {
        return new LIRStandardPipeline().run(graph);
    }

    /**
     * Convenience method to run the standard pipeline with multiple iterations.
     *
     * <p>Use this when you need more aggressive optimization.
     *
     * @param graph the input LIR graph
     * @param iterations number of iterations to run
     * @return the optimized LIR graph
     */
    public static LIRGraph optimize(LIRGraph graph, int iterations) {
        return new LIRStandardPipeline().withIterations(iterations).run(graph);
    }

    /**
     * Returns the list of passes in this pipeline.
     *
     * @return list of passes
     */
    public List<LIRPass> getPasses() {
        return new ArrayList<>(passes);
    }

    /**
     * Returns a string description of the pipeline.
     *
     * @return description
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LIRStandardPipeline (").append(iterations).append(" iteration");
        if (iterations > 1) sb.append("s");
        sb.append("):\n");

        for (int i = 0; i < passes.size(); i++) {
            sb.append("  ").append(i + 1).append(". ").append(passes.get(i).name()).append("\n");
        }

        return sb.toString();
    }
}

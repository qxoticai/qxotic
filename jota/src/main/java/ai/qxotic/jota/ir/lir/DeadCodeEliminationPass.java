package ai.qxotic.jota.ir.lir;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Dead Code Elimination pass that removes unused ScalarLet definitions from LIR graphs.
 *
 * <p>A ScalarLet is considered "dead" if its name is never referenced by any ScalarRef within its
 * scope. For example:
 *
 * <pre>
 * // Before:
 * %dead = multiply fp32 %a, %b
 * %alive = add fp32 %x, %y
 * store %out[0], %alive
 *
 * // After DCE:
 * %alive = add fp32 %x, %y
 * store %out[0], %alive
 * </pre>
 *
 * <p>This pass operates in two phases:
 *
 * <ol>
 *   <li>Analysis: Collect all ScalarLet definitions and ScalarRef references
 *   <li>Elimination: Remove ScalarLet nodes with no references
 * </ol>
 *
 * <p>Scope rules: ScalarLet bindings are only visible within the same Block after their definition.
 * References can appear in subsequent statements or nested structures.
 */
public final class DeadCodeEliminationPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        // Phase 1: Find all dead ScalarLets
        Set<String> defined = new HashSet<>();
        Set<String> referenced = new HashSet<>();
        analyzeReferences(graph.body(), defined, referenced);

        // Find dead lets: defined but never referenced
        Set<String> deadLets = new HashSet<>(defined);
        deadLets.removeAll(referenced);

        if (deadLets.isEmpty()) {
            return graph; // No dead code
        }

        // Phase 2: Remove dead lets
        LIRNode newBody = removeDeadLets(graph.body(), deadLets);
        return new LIRGraph(graph.inputs(), graph.outputs(), newBody);
    }

    @Override
    public String name() {
        return "DeadCodeElimination";
    }

    /**
     * Analyzes a node to collect ScalarLet definitions and ScalarRef references.
     *
     * @param node the node to analyze
     * @param defined set to which ScalarLet names are added
     * @param referenced set to which ScalarRef names are added
     */
    private void analyzeReferences(LIRNode node, Set<String> defined, Set<String> referenced) {
        switch (node) {
            case ScalarLet let -> {
                // Record the definition
                defined.add(let.name());
                // Analyze the value expression for references
                collectRefsInExpr(let.value(), referenced);
            }
            case Block block -> {
                for (LIRNode stmt : block.statements()) {
                    analyzeReferences(stmt, defined, referenced);
                }
            }
            case Loop loop -> analyzeReferences(loop.body(), defined, referenced);
            case StructuredFor structuredFor -> {
                for (LoopIterArg arg : structuredFor.iterArgs()) {
                    collectRefsInExpr(arg.init(), referenced);
                }
                analyzeReferences(structuredFor.body(), defined, referenced);
            }
            case TiledLoop tiled -> analyzeReferences(tiled.body(), defined, referenced);
            case LoopNest nest -> analyzeReferences(nest.body(), defined, referenced);
            case Store store -> collectRefsInExpr(store.value(), referenced);
            case Yield yield -> {
                for (ScalarExpr value : yield.values()) {
                    collectRefsInExpr(value, referenced);
                }
            }
            default -> {
                // Other nodes don't define or reference scalar names
            }
        }
    }

    /**
     * Collects all ScalarRef names referenced in a scalar expression.
     *
     * @param expr the expression to analyze
     * @param referenced set to which ScalarRef names are added
     */
    private void collectRefsInExpr(ScalarExpr expr, Set<String> referenced) {
        switch (expr) {
            case ScalarRef ref -> referenced.add(ref.name());
            case ScalarBinary bin -> {
                collectRefsInExpr(bin.left(), referenced);
                collectRefsInExpr(bin.right(), referenced);
            }
            case ScalarUnary un -> collectRefsInExpr(un.input(), referenced);
            case ScalarTernary ter -> {
                collectRefsInExpr(ter.condition(), referenced);
                collectRefsInExpr(ter.trueValue(), referenced);
                collectRefsInExpr(ter.falseValue(), referenced);
            }
            case ScalarCast cast -> collectRefsInExpr(cast.input(), referenced);
            case ScalarLoad load -> collectRefsInIndex(load.offset(), referenced);
            case ScalarFromIndex sfi -> collectRefsInIndex(sfi.index(), referenced);
            default -> {
                // ScalarLiteral, ScalarInput - no references
            }
        }
    }

    /**
     * Collects all ScalarRef names referenced in an index expression.
     *
     * <p>Note: Index expressions don't directly contain ScalarRefs, but this is here for
     * completeness in case future extensions allow it.
     *
     * @param expr the index expression to analyze
     * @param referenced set to which names would be added (currently unused)
     */
    private void collectRefsInIndex(IndexExpr expr, Set<String> referenced) {
        switch (expr) {
            case IndexBinary bin -> {
                collectRefsInIndex(bin.left(), referenced);
                collectRefsInIndex(bin.right(), referenced);
            }
            default -> {
                // IndexConst, IndexVar - no references
            }
        }
    }

    /**
     * Removes dead ScalarLet nodes from the graph.
     *
     * @param node the node to process
     * @param deadLets set of dead ScalarLet names to remove
     * @return the transformed node (may be the same instance if no changes)
     */
    private LIRNode removeDeadLets(LIRNode node, Set<String> deadLets) {
        return switch (node) {
            case ScalarLet let -> {
                if (deadLets.contains(let.name())) {
                    // Remove this dead let
                    yield null;
                }
                yield let;
            }
            case Block block -> {
                List<LIRNode> newStmts = new java.util.ArrayList<>();
                for (LIRNode stmt : block.statements()) {
                    LIRNode newStmt = removeDeadLets(stmt, deadLets);
                    if (newStmt != null) {
                        newStmts.add(newStmt);
                    }
                }
                if (newStmts.size() == block.statements().size()) {
                    // No statements removed
                    yield block;
                }
                yield newStmts.isEmpty()
                        ? null
                        : (newStmts.size() == 1 ? newStmts.get(0) : new Block(newStmts));
            }
            case Loop loop -> {
                LIRNode newBody = removeDeadLets(loop.body(), deadLets);
                if (newBody == loop.body()) {
                    yield loop;
                }
                if (newBody == null) {
                    // Empty loop body - this shouldn't happen in practice
                    // but we handle it by creating an empty block
                    yield new Loop(
                            loop.indexName(),
                            loop.bound(),
                            loop.isParallel(),
                            new Block(List.of()));
                }
                yield new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
            }
            case StructuredFor structuredFor -> {
                LIRNode newBody = removeDeadLets(structuredFor.body(), deadLets);
                if (newBody == structuredFor.body()) {
                    yield structuredFor;
                }
                if (newBody == null) {
                    yield structuredFor;
                }
                yield new StructuredFor(
                        structuredFor.indexName(),
                        structuredFor.lowerBound(),
                        structuredFor.upperBound(),
                        structuredFor.step(),
                        structuredFor.iterArgs(),
                        newBody);
            }
            case TiledLoop tiled -> {
                LIRNode newBody = removeDeadLets(tiled.body(), deadLets);
                if (newBody == tiled.body()) {
                    yield tiled;
                }
                if (newBody == null) {
                    yield new TiledLoop(
                            tiled.outerName(),
                            tiled.innerName(),
                            tiled.totalBound(),
                            tiled.tileSize(),
                            new Block(List.of()));
                }
                yield new TiledLoop(
                        tiled.outerName(),
                        tiled.innerName(),
                        tiled.totalBound(),
                        tiled.tileSize(),
                        newBody);
            }
            case LoopNest nest -> {
                LIRNode newBody = removeDeadLets(nest.body(), deadLets);
                if (newBody == nest.body()) {
                    yield nest;
                }
                if (newBody == null) {
                    yield new LoopNest(nest.loops(), new Block(List.of()));
                }
                yield new LoopNest(nest.loops(), newBody);
            }
            default -> node;
        };
    }
}

package ai.qxotic.jota.ir.lir;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Loop-invariant code motion pass that hoists computations that don't depend on loop indices
 * outside the loop.
 *
 * <p>For example, transforms:
 *
 * <pre>
 * for %i in [0, N) {
 *   %0 = multiply fp32 %scalar0, %scalar1
 *   update.acc %acc, %0
 * }
 * </pre>
 *
 * Into:
 *
 * <pre>
 * %hoisted0 = multiply fp32 %scalar0, %scalar1
 * for %i in [0, N) {
 *   update.acc %acc, %hoisted0
 * }
 * </pre>
 *
 * <p>This pass identifies loop-invariant scalar expressions and creates ScalarLet definitions
 * before the loop.
 */
public final class LoopInvariantHoistingPass implements LIRPass {

    private int nextHoistedId = 0;

    @Override
    public LIRGraph run(LIRGraph graph) {
        // Find existing hoisted names to avoid collisions
        nextHoistedId = findMaxHoistedId(graph.body()) + 1;
        LIRNode newBody = hoistInNode(graph.body(), Set.of());
        if (newBody == graph.body()) {
            return graph;
        }
        return new LIRGraph(graph.inputs(), graph.outputs(), newBody);
    }

    /** Recursively scans the node tree for ScalarLet nodes with names starting with "hoisted". */
    private int findMaxHoistedId(LIRNode node) {
        return switch (node) {
            case ScalarLet let -> {
                String name = let.name();
                if (name.startsWith("hoisted")) {
                    try {
                        int id = Integer.parseInt(name.substring(7));
                        yield id;
                    } catch (NumberFormatException e) {
                        yield -1;
                    }
                }
                yield -1;
            }
            case Block block -> {
                int max = -1;
                for (LIRNode stmt : block.statements()) {
                    max = Math.max(max, findMaxHoistedId(stmt));
                }
                yield max;
            }
            case Loop loop -> findMaxHoistedId(loop.body());
            case StructuredFor structuredFor -> findMaxHoistedId(structuredFor.body());
            case TiledLoop tiled -> findMaxHoistedId(tiled.body());
            case LoopNest nest -> findMaxHoistedId(nest.body());
            default -> -1;
        };
    }

    @Override
    public String name() {
        return "LoopInvariantHoisting";
    }

    /**
     * Process a node, hoisting loop-invariant expressions where possible.
     *
     * @param node the node to process
     * @param boundIndices set of index variable names that are currently bound (from enclosing
     *     loops)
     * @return the transformed node (may be the same instance if no hoisting occurred)
     */
    private LIRNode hoistInNode(LIRNode node, Set<String> boundIndices) {
        return switch (node) {
            case Loop loop -> hoistInLoop(loop, boundIndices);
            case StructuredFor structuredFor -> hoistInStructuredFor(structuredFor, boundIndices);
            case TiledLoop tiled -> hoistInTiledLoop(tiled, boundIndices);
            case LoopNest nest -> hoistInLoopNest(nest, boundIndices);
            case Block block -> hoistInBlock(block, boundIndices);
            case ScalarLet let -> hoistInScalarLet(let, boundIndices);
            case Store store -> hoistInStore(store, boundIndices);
            case Yield yield -> hoistInYield(yield, boundIndices);
            default -> node; // Other nodes don't contain loops or don't need hoisting
        };
    }

    private LIRNode hoistInLoop(Loop loop, Set<String> boundIndices) {
        // Add this loop's index to the set of bound indices
        Set<String> innerBound = new HashSet<>(boundIndices);
        innerBound.add(loop.indexName());

        // First, recursively process the body
        LIRNode processedBody = hoistInNode(loop.body(), innerBound);

        // Find loop-invariant expressions in the processed body
        List<HoistCandidate> candidates = findHoistCandidates(processedBody, loop.indexName());

        // Hoist the candidates - substitute references in the loop body
        LIRNode hoistedBody = processedBody;
        for (HoistCandidate candidate : candidates) {
            hoistedBody = substituteExpr(hoistedBody, candidate.expr(), candidate.ref());
        }

        // Also find and extract loop-invariant ScalarLets from the body
        // This handles ScalarLets created by inner loop processing that can be hoisted further
        List<ScalarLet> hoistedLets = new ArrayList<>();
        hoistedBody =
                extractInvariantLets(hoistedBody, loop.indexName(), Set.<String>of(), hoistedLets);

        // Clean up trivial ScalarLets (where value is now just a ScalarRef)
        hoistedBody = cleanupTrivialLets(hoistedBody);

        // If nothing changed, return original loop
        if (candidates.isEmpty() && hoistedLets.isEmpty() && hoistedBody == processedBody) {
            if (processedBody == loop.body()) {
                return loop;
            }
            return new Loop(loop.indexName(), loop.bound(), loop.isParallel(), processedBody);
        }

        Loop newLoop = new Loop(loop.indexName(), loop.bound(), loop.isParallel(), hoistedBody);

        // If nothing was hoisted, just return the new loop
        if (candidates.isEmpty() && hoistedLets.isEmpty()) {
            return newLoop;
        }

        // Create Block with hoisted definitions followed by the loop
        List<LIRNode> statements = new ArrayList<>();
        // First add new ScalarLets for hoisted expressions
        for (HoistCandidate candidate : candidates) {
            statements.add(new ScalarLet(candidate.name(), candidate.expr()));
        }
        // Then add existing ScalarLets that were moved out
        statements.addAll(hoistedLets);
        statements.add(newLoop);

        return new Block(statements);
    }

    private LIRNode hoistInStructuredFor(StructuredFor loop, Set<String> boundIndices) {
        Set<String> innerBound = new HashSet<>(boundIndices);
        innerBound.add(loop.indexName());

        LIRNode processedBody = hoistInNode(loop.body(), innerBound);

        Set<String> iterArgNames = new HashSet<>();
        for (LoopIterArg arg : loop.iterArgs()) {
            iterArgNames.add(arg.name());
        }

        List<HoistCandidate> candidates =
                findHoistCandidates(processedBody, loop.indexName(), iterArgNames);
        LIRNode hoistedBody = processedBody;
        for (HoistCandidate candidate : candidates) {
            hoistedBody = substituteExpr(hoistedBody, candidate.expr(), candidate.ref());
        }

        List<ScalarLet> hoistedLets = new ArrayList<>();
        hoistedBody =
                extractInvariantLets(hoistedBody, loop.indexName(), iterArgNames, hoistedLets);
        hoistedBody = cleanupTrivialLets(hoistedBody);

        if (candidates.isEmpty() && hoistedLets.isEmpty() && hoistedBody == processedBody) {
            if (processedBody == loop.body()) {
                return loop;
            }
            return new StructuredFor(
                    loop.indexName(),
                    loop.lowerBound(),
                    loop.upperBound(),
                    loop.step(),
                    loop.iterArgs(),
                    processedBody);
        }

        StructuredFor newLoop =
                new StructuredFor(
                        loop.indexName(),
                        loop.lowerBound(),
                        loop.upperBound(),
                        loop.step(),
                        loop.iterArgs(),
                        hoistedBody);

        if (candidates.isEmpty() && hoistedLets.isEmpty()) {
            return newLoop;
        }

        List<LIRNode> statements = new ArrayList<>();
        for (HoistCandidate candidate : candidates) {
            statements.add(new ScalarLet(candidate.name(), candidate.expr()));
        }
        statements.addAll(hoistedLets);
        statements.add(newLoop);

        return new Block(statements);
    }

    private LIRNode hoistInTiledLoop(TiledLoop tiled, Set<String> boundIndices) {
        Set<String> innerBound = new HashSet<>(boundIndices);
        innerBound.add(tiled.outerName());
        innerBound.add(tiled.innerName());

        LIRNode processedBody = hoistInNode(tiled.body(), innerBound);
        if (processedBody == tiled.body()) {
            return tiled;
        }
        return new TiledLoop(
                tiled.outerName(),
                tiled.innerName(),
                tiled.totalBound(),
                tiled.tileSize(),
                processedBody);
    }

    private LIRNode hoistInLoopNest(LoopNest nest, Set<String> boundIndices) {
        Set<String> innerBound = new HashSet<>(boundIndices);
        for (Loop loop : nest.loops()) {
            innerBound.add(loop.indexName());
        }

        LIRNode processedBody = hoistInNode(nest.body(), innerBound);
        if (processedBody == nest.body()) {
            return nest;
        }
        return new LoopNest(nest.loops(), processedBody);
    }

    private LIRNode hoistInBlock(Block block, Set<String> boundIndices) {
        List<LIRNode> newStatements = new ArrayList<>(block.statements().size());
        boolean changed = false;

        for (LIRNode stmt : block.statements()) {
            LIRNode newStmt = hoistInNode(stmt, boundIndices);
            newStatements.add(newStmt);
            if (newStmt != stmt) {
                changed = true;
            }
        }

        if (!changed) {
            return block;
        }
        return new Block(newStatements);
    }

    private LIRNode hoistInScalarLet(ScalarLet let, Set<String> boundIndices) {
        // ScalarLet is just a definition, no body to process
        return let;
    }

    /**
     * Extracts ScalarLets whose values are loop-invariant from the node tree. The extracted lets
     * are added to the hoistedLets list and removed from the tree.
     */
    private LIRNode extractInvariantLets(
            LIRNode node, String loopIndex, Set<String> blockedRefs, List<ScalarLet> hoistedLets) {
        return switch (node) {
            case Block block -> {
                List<LIRNode> newStmts = new ArrayList<>();
                for (LIRNode stmt : block.statements()) {
                    if (stmt instanceof ScalarLet let
                            && !dependsOnIndex(let.value(), loopIndex)
                            && !dependsOnRefs(let.value(), blockedRefs)) {
                        // This ScalarLet is loop-invariant, extract it
                        hoistedLets.add(let);
                    } else {
                        // Recursively process
                        LIRNode processed =
                                extractInvariantLets(stmt, loopIndex, blockedRefs, hoistedLets);
                        if (processed != null) {
                            newStmts.add(processed);
                        }
                    }
                }
                yield newStmts.isEmpty()
                        ? null
                        : (newStmts.size() == 1 ? newStmts.get(0) : new Block(newStmts));
            }
            case Loop loop -> {
                LIRNode newBody =
                        extractInvariantLets(loop.body(), loopIndex, blockedRefs, hoistedLets);
                if (newBody == loop.body()) yield loop;
                yield new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
            }
            case StructuredFor structuredFor -> {
                LIRNode newBody =
                        extractInvariantLets(
                                structuredFor.body(), loopIndex, blockedRefs, hoistedLets);
                if (newBody == structuredFor.body()) yield structuredFor;
                yield new StructuredFor(
                        structuredFor.indexName(),
                        structuredFor.lowerBound(),
                        structuredFor.upperBound(),
                        structuredFor.step(),
                        structuredFor.iterArgs(),
                        newBody);
            }
            default -> node;
        };
    }

    private boolean dependsOnRefs(ScalarExpr expr, Set<String> refNames) {
        if (refNames.isEmpty()) {
            return false;
        }
        return switch (expr) {
            case ScalarRef ref -> refNames.contains(ref.name());
            case ScalarLiteral ignored -> false;
            case ScalarInput ignored -> false;
            case ScalarFromIndex sfi -> dependsOnRefsInIndex(sfi.index(), refNames);
            case ScalarLoad load -> dependsOnRefsInIndex(load.offset(), refNames);
            case ScalarUnary unary -> dependsOnRefs(unary.input(), refNames);
            case ScalarBinary binary ->
                    dependsOnRefs(binary.left(), refNames)
                            || dependsOnRefs(binary.right(), refNames);
            case ScalarTernary ternary ->
                    dependsOnRefs(ternary.condition(), refNames)
                            || dependsOnRefs(ternary.trueValue(), refNames)
                            || dependsOnRefs(ternary.falseValue(), refNames);
            case ScalarCast cast -> dependsOnRefs(cast.input(), refNames);
        };
    }

    private boolean dependsOnRefsInIndex(IndexExpr expr, Set<String> refNames) {
        return switch (expr) {
            case IndexConst ignored -> false;
            case IndexVar ignored -> false;
            case IndexBinary binary ->
                    dependsOnRefsInIndex(binary.left(), refNames)
                            || dependsOnRefsInIndex(binary.right(), refNames);
        };
    }

    /**
     * Removes trivial ScalarLets where the value is just a ScalarRef, and renames all references.
     */
    private LIRNode cleanupTrivialLets(LIRNode node) {
        // First, collect trivial lets (name -> target ref name)
        Map<String, String> renames = new HashMap<>();
        collectTrivialLets(node, renames);

        if (renames.isEmpty()) {
            return node;
        }

        // Apply renames and remove trivial lets
        return applyRenamesAndRemoveTrivialLets(node, renames);
    }

    private void collectTrivialLets(LIRNode node, Map<String, String> renames) {
        switch (node) {
            case ScalarLet let -> {
                if (let.value() instanceof ScalarRef ref) {
                    // This let is trivial: %let.name = %ref.name
                    // We want to rename all uses of %let.name to %ref.name
                    renames.put(let.name(), ref.name());
                }
            }
            case Block block -> {
                for (LIRNode stmt : block.statements()) {
                    collectTrivialLets(stmt, renames);
                }
            }
            case Loop loop -> collectTrivialLets(loop.body(), renames);
            case StructuredFor structuredFor -> collectTrivialLets(structuredFor.body(), renames);
            default -> {}
        }
    }

    private LIRNode applyRenamesAndRemoveTrivialLets(LIRNode node, Map<String, String> renames) {
        return switch (node) {
            case ScalarLet let -> {
                if (let.value() instanceof ScalarRef) {
                    // Remove this trivial let
                    yield null;
                }
                yield let;
            }
            case Block block -> {
                List<LIRNode> newStmts = new ArrayList<>();
                for (LIRNode stmt : block.statements()) {
                    LIRNode newStmt = applyRenamesAndRemoveTrivialLets(stmt, renames);
                    if (newStmt != null) {
                        newStmts.add(newStmt);
                    }
                }
                yield new Block(newStmts);
            }
            case Loop loop -> {
                LIRNode newBody = applyRenamesAndRemoveTrivialLets(loop.body(), renames);
                yield new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
            }
            case StructuredFor structuredFor -> {
                LIRNode newBody = applyRenamesAndRemoveTrivialLets(structuredFor.body(), renames);
                yield new StructuredFor(
                        structuredFor.indexName(),
                        structuredFor.lowerBound(),
                        structuredFor.upperBound(),
                        structuredFor.step(),
                        structuredFor.iterArgs(),
                        newBody);
            }
            case Store store -> {
                ScalarExpr newValue = renameRefs(store.value(), renames);
                yield new Store(store.buffer(), store.offset(), newValue);
            }
            case Yield yield -> {
                java.util.List<ScalarExpr> newValues =
                        new java.util.ArrayList<>(yield.values().size());
                for (ScalarExpr value : yield.values()) {
                    newValues.add(renameRefs(value, renames));
                }
                yield new Yield(newValues);
            }
            default -> node;
        };
    }

    private ScalarExpr renameRefs(ScalarExpr expr, Map<String, String> renames) {
        return switch (expr) {
            case ScalarRef ref -> {
                String target = resolveRename(ref.name(), renames);
                if (target.equals(ref.name())) {
                    yield ref;
                }
                yield new ScalarRef(target, ref.dataType());
            }
            case ScalarBinary bin -> {
                ScalarExpr left = renameRefs(bin.left(), renames);
                ScalarExpr right = renameRefs(bin.right(), renames);
                if (left == bin.left() && right == bin.right()) yield bin;
                yield new ScalarBinary(bin.op(), left, right);
            }
            case ScalarUnary un -> {
                ScalarExpr input = renameRefs(un.input(), renames);
                if (input == un.input()) yield un;
                yield new ScalarUnary(un.op(), input);
            }
            case ScalarCast cast -> {
                ScalarExpr input = renameRefs(cast.input(), renames);
                if (input == cast.input()) yield cast;
                yield new ScalarCast(input, cast.targetType());
            }
            case ScalarTernary ter -> {
                ScalarExpr cond = renameRefs(ter.condition(), renames);
                ScalarExpr trueVal = renameRefs(ter.trueValue(), renames);
                ScalarExpr falseVal = renameRefs(ter.falseValue(), renames);
                if (cond == ter.condition()
                        && trueVal == ter.trueValue()
                        && falseVal == ter.falseValue()) yield ter;
                yield new ScalarTernary(cond, trueVal, falseVal);
            }
            default -> expr;
        };
    }

    /** Follows the rename chain to the final target. */
    private String resolveRename(String name, Map<String, String> renames) {
        String current = name;
        while (renames.containsKey(current)) {
            current = renames.get(current);
        }
        return current;
    }

    private LIRNode hoistInStore(Store store, Set<String> boundIndices) {
        // Stores don't contain nested loops, no hoisting needed
        return store;
    }

    private LIRNode hoistInYield(Yield yield, Set<String> boundIndices) {
        return yield;
    }

    /**
     * Finds scalar expressions in the body that don't depend on the given loop index and are worth
     * hoisting.
     */
    private List<HoistCandidate> findHoistCandidates(LIRNode body, String loopIndex) {
        List<HoistCandidate> candidates = new ArrayList<>();
        Set<ScalarExpr> seen = new HashSet<>();
        collectHoistCandidates(body, loopIndex, candidates, seen);
        return candidates;
    }

    private List<HoistCandidate> findHoistCandidates(
            LIRNode body, String loopIndex, Set<String> blockedRefs) {
        if (blockedRefs.isEmpty()) {
            return findHoistCandidates(body, loopIndex);
        }
        List<HoistCandidate> candidates = new ArrayList<>();
        Set<ScalarExpr> seen = new HashSet<>();
        collectHoistCandidates(body, loopIndex, blockedRefs, candidates, seen);
        return candidates;
    }

    private void collectHoistCandidates(
            LIRNode node, String loopIndex, List<HoistCandidate> candidates, Set<ScalarExpr> seen) {
        switch (node) {
            case Store store -> {
                maybeAddCandidate(store.value(), loopIndex, candidates, seen);
            }
            case Yield yield -> {
                for (ScalarExpr value : yield.values()) {
                    maybeAddCandidate(value, loopIndex, candidates, seen);
                }
            }
            case Loop loop -> {
                // Don't descend into nested loops for hoisting candidates
                // (they will be processed separately)
            }
            case StructuredFor structuredFor -> {
                // Don't descend into nested loops for hoisting candidates
            }
            case TiledLoop tiled -> {
                // Don't descend into nested loops
            }
            case Block block -> {
                for (LIRNode stmt : block.statements()) {
                    collectHoistCandidates(stmt, loopIndex, candidates, seen);
                }
            }
            case ScalarLet let -> {
                // ScalarLets are already hoisted definitions - don't re-hoist their values.
                // If this ScalarLet is inside the loop and its value is loop-invariant,
                // it will be moved out as part of the Block restructuring.
            }
            default -> {}
        }
    }

    private void collectHoistCandidates(
            LIRNode node,
            String loopIndex,
            Set<String> blockedRefs,
            List<HoistCandidate> candidates,
            Set<ScalarExpr> seen) {
        switch (node) {
            case Store store -> {
                maybeAddCandidate(store.value(), loopIndex, blockedRefs, candidates, seen);
            }
            case Yield yield -> {
                for (ScalarExpr value : yield.values()) {
                    maybeAddCandidate(value, loopIndex, blockedRefs, candidates, seen);
                }
            }
            case Loop loop -> {
                // Don't descend into nested loops
            }
            case StructuredFor structuredFor -> {
                // Don't descend into nested loops
            }
            case TiledLoop tiled -> {
                // Don't descend into nested loops
            }
            case Block block -> {
                for (LIRNode stmt : block.statements()) {
                    collectHoistCandidates(stmt, loopIndex, blockedRefs, candidates, seen);
                }
            }
            case ScalarLet let -> {
                // ScalarLets are already hoisted definitions
            }
            default -> {}
        }
    }

    private void maybeAddCandidate(
            ScalarExpr expr,
            String loopIndex,
            List<HoistCandidate> candidates,
            Set<ScalarExpr> seen) {
        // Skip if already processed
        if (seen.contains(expr)) {
            return;
        }
        seen.add(expr);

        // Don't hoist trivial expressions
        if (isTrivial(expr)) {
            return;
        }

        // Check if expression is loop-invariant
        if (!dependsOnIndex(expr, loopIndex)) {
            String name = "hoisted" + nextHoistedId++;
            ScalarRef ref = new ScalarRef(name, expr.dataType());
            candidates.add(new HoistCandidate(name, expr, ref));
        } else {
            // Recursively check sub-expressions
            collectSubExprCandidates(expr, loopIndex, candidates, seen);
        }
    }

    private void maybeAddCandidate(
            ScalarExpr expr,
            String loopIndex,
            Set<String> blockedRefs,
            List<HoistCandidate> candidates,
            Set<ScalarExpr> seen) {
        if (seen.contains(expr)) {
            return;
        }
        seen.add(expr);

        if (isTrivial(expr)) {
            return;
        }

        if (!dependsOnIndex(expr, loopIndex) && !dependsOnRefs(expr, blockedRefs)) {
            String name = "hoisted" + nextHoistedId++;
            ScalarRef ref = new ScalarRef(name, expr.dataType());
            candidates.add(new HoistCandidate(name, expr, ref));
        } else {
            collectSubExprCandidates(expr, loopIndex, blockedRefs, candidates, seen);
        }
    }

    private void collectSubExprCandidates(
            ScalarExpr expr,
            String loopIndex,
            List<HoistCandidate> candidates,
            Set<ScalarExpr> seen) {
        switch (expr) {
            case ScalarBinary binary -> {
                maybeAddCandidate(binary.left(), loopIndex, candidates, seen);
                maybeAddCandidate(binary.right(), loopIndex, candidates, seen);
            }
            case ScalarUnary unary -> {
                maybeAddCandidate(unary.input(), loopIndex, candidates, seen);
            }
            case ScalarTernary ternary -> {
                maybeAddCandidate(ternary.condition(), loopIndex, candidates, seen);
                maybeAddCandidate(ternary.trueValue(), loopIndex, candidates, seen);
                maybeAddCandidate(ternary.falseValue(), loopIndex, candidates, seen);
            }
            case ScalarCast cast -> {
                maybeAddCandidate(cast.input(), loopIndex, candidates, seen);
            }
            default -> {}
        }
    }

    private void collectSubExprCandidates(
            ScalarExpr expr,
            String loopIndex,
            Set<String> blockedRefs,
            List<HoistCandidate> candidates,
            Set<ScalarExpr> seen) {
        switch (expr) {
            case ScalarBinary binary -> {
                maybeAddCandidate(binary.left(), loopIndex, blockedRefs, candidates, seen);
                maybeAddCandidate(binary.right(), loopIndex, blockedRefs, candidates, seen);
            }
            case ScalarUnary unary -> {
                maybeAddCandidate(unary.input(), loopIndex, blockedRefs, candidates, seen);
            }
            case ScalarTernary ternary -> {
                maybeAddCandidate(ternary.condition(), loopIndex, blockedRefs, candidates, seen);
                maybeAddCandidate(ternary.trueValue(), loopIndex, blockedRefs, candidates, seen);
                maybeAddCandidate(ternary.falseValue(), loopIndex, blockedRefs, candidates, seen);
            }
            case ScalarCast cast -> {
                maybeAddCandidate(cast.input(), loopIndex, blockedRefs, candidates, seen);
            }
            default -> {}
        }
    }

    /** Returns true if the expression is too trivial to be worth hoisting. */
    private boolean isTrivial(ScalarExpr expr) {
        return switch (expr) {
            case ScalarLiteral ignored -> true;
            case ScalarInput ignored -> true;
            case ScalarRef ignored -> true;
            default -> false;
        };
    }

    /** Returns true if the expression depends on the given index variable. */
    private boolean dependsOnIndex(ScalarExpr expr, String indexName) {
        return switch (expr) {
            case ScalarLiteral ignored -> false;
            case ScalarInput ignored -> false;
            case ScalarRef ignored -> false;
            case ScalarFromIndex sfi -> dependsOnIndexExpr(sfi.index(), indexName);
            case ScalarLoad load -> dependsOnIndexExpr(load.offset(), indexName);
            case ScalarUnary unary -> dependsOnIndex(unary.input(), indexName);
            case ScalarBinary binary ->
                    dependsOnIndex(binary.left(), indexName)
                            || dependsOnIndex(binary.right(), indexName);
            case ScalarTernary ternary ->
                    dependsOnIndex(ternary.condition(), indexName)
                            || dependsOnIndex(ternary.trueValue(), indexName)
                            || dependsOnIndex(ternary.falseValue(), indexName);
            case ScalarCast cast -> dependsOnIndex(cast.input(), indexName);
        };
    }

    /** Returns true if the index expression depends on the given index variable. */
    private boolean dependsOnIndexExpr(IndexExpr expr, String indexName) {
        return switch (expr) {
            case IndexConst ignored -> false;
            case IndexVar var -> var.name().equals(indexName);
            case IndexBinary binary ->
                    dependsOnIndexExpr(binary.left(), indexName)
                            || dependsOnIndexExpr(binary.right(), indexName);
        };
    }

    /** Substitutes all occurrences of the given expression with the reference in the node tree. */
    private LIRNode substituteExpr(LIRNode node, ScalarExpr target, ScalarRef replacement) {
        return new ExprSubstituter(target, replacement).substitute(node);
    }

    private record HoistCandidate(String name, ScalarExpr expr, ScalarRef ref) {}

    /** Helper class to substitute scalar expressions. */
    private static class ExprSubstituter extends LIRRewriter {
        private final ScalarExpr target;
        private final ScalarRef replacement;

        ExprSubstituter(ScalarExpr target, ScalarRef replacement) {
            this.target = target;
            this.replacement = replacement;
        }

        LIRNode substitute(LIRNode node) {
            return node.accept(this);
        }

        @Override
        public LIRNode visitScalarBinary(ScalarBinary node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarBinary(node);
        }

        @Override
        public LIRNode visitScalarUnary(ScalarUnary node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarUnary(node);
        }

        @Override
        public LIRNode visitScalarTernary(ScalarTernary node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarTernary(node);
        }

        @Override
        public LIRNode visitScalarCast(ScalarCast node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarCast(node);
        }

        @Override
        public LIRNode visitScalarLoad(ScalarLoad node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarLoad(node);
        }

        @Override
        public LIRNode visitScalarFromIndex(ScalarFromIndex node) {
            if (node == target) {
                return replacement;
            }
            return super.visitScalarFromIndex(node);
        }
    }
}

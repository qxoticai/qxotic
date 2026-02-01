package ai.qxotic.jota.ir.lir;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pass that simplifies index expressions using algebraic rules and range information from loop
 * bounds. Leverages the existing simplification logic in {@link IndexBinary} and {@link
 * IndexExpr#simplify()}, and adds range-based simplifications.
 *
 * <p>Simplifications include:
 *
 * <ul>
 *   <li>Identity elimination: x + 0 → x, x * 1 → x, x / 1 → x
 *   <li>Zero propagation: x * 0 → 0, 0 / x → 0
 *   <li>Constant folding: 3 + 4 → 7
 *   <li>Strength reduction: x * 2^n → x << n, x / 2^n → x >> n
 *   <li>Modulo optimization: x % 2^n → x & (2^n - 1)
 *   <li>Range-based simplifications (using loop bounds):
 *       <ul>
 *         <li>x / C → 0 when x < C (e.g., i2 / 3 when i2 ∈ [0, 3))
 *         <li>x % C → x when x < C (e.g., i2 % 3 when i2 ∈ [0, 3))
 *         <li>0 & x → 0
 *       </ul>
 * </ul>
 */
public final class IndexSimplificationPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        return new IndexSimplificationRewriter(Map.of()).rewrite(graph);
    }

    @Override
    public String name() {
        return "IndexSimplification";
    }

    /**
     * Extracts a constant bound value from an index expression.
     *
     * @param expr the index expression
     * @return the constant value if expr is a constant, -1 otherwise
     */
    private static long extractConstantBound(IndexExpr expr) {
        if (expr instanceof IndexConst c) {
            return c.value();
        }
        return -1;
    }

    /**
     * Gets the upper bound for a variable if it's in the ranges map.
     *
     * @param expr the index expression
     * @param ranges map of variable names to their upper bounds
     * @return the upper bound, or -1 if not found
     */
    private static long getVarRange(IndexExpr expr, Map<String, Long> ranges) {
        if (expr instanceof IndexVar v) {
            return ranges.getOrDefault(v.name(), -1L);
        }
        return -1;
    }

    /**
     * Checks if an index expression is less than a constant (conservatively).
     *
     * @param expr the index expression
     * @param constant the constant to compare against
     * @param ranges map of variable names to their upper bounds
     * @return true if we can prove expr < constant
     */
    private static boolean isLessThan(IndexExpr expr, long constant, Map<String, Long> ranges) {
        if (constant <= 0) {
            return false;
        }

        // Direct constant comparison
        if (expr instanceof IndexConst c) {
            return c.value() < constant;
        }

        // Variable with known range
        if (expr instanceof IndexVar v) {
            Long range = ranges.get(v.name());
            if (range != null && range <= constant) {
                return true;
            }
        }

        // For other expressions, be conservative
        return false;
    }

    /**
     * Attempts additional simplifications based on variable ranges.
     *
     * @param bin the binary expression to simplify
     * @param ranges map of variable names to their upper bounds
     * @return simplified expression, or the original if no simplification possible
     */
    private static IndexExpr simplifyWithRanges(IndexBinary bin, Map<String, Long> ranges) {
        return switch (bin.op()) {
            case DIVIDE -> {
                // x / C → 0 when x < C and x >= 0
                // This handles the case: i2 / 3 when i2 ∈ [0, 3)
                if (bin.right().isConstant() && bin.right().constantValue() > 0) {
                    long divisor = bin.right().constantValue();
                    if (isLessThan(bin.left(), divisor, ranges)) {
                        yield IndexConst.ZERO;
                    }
                }
                yield bin;
            }
            case MODULO -> {
                // x % C → x when 0 <= x < C
                // This handles the case: i2 % 3 when i2 ∈ [0, 3)
                if (bin.right().isConstant() && bin.right().constantValue() > 0) {
                    long mod = bin.right().constantValue();
                    if (isLessThan(bin.left(), mod, ranges)) {
                        yield bin.left();
                    }
                }
                yield bin;
            }
            case BITWISE_AND -> {
                // 0 & x → 0
                if (bin.left().isZero()) {
                    yield IndexConst.ZERO;
                }
                // x & 0 → 0
                if (bin.right().isZero()) {
                    yield IndexConst.ZERO;
                }
                yield bin;
            }
            default -> bin;
        };
    }

    private static final class IndexSimplificationRewriter extends LIRRewriter {
        private final Map<String, Long> varRanges;

        IndexSimplificationRewriter(Map<String, Long> ranges) {
            this.varRanges = ranges;
        }

        @Override
        public LIRNode visitLoop(Loop loop) {
            // Extract the bound if it's a constant
            long bound = extractConstantBound(loop.bound());

            if (bound > 0) {
                // Create extended context with this variable's range
                Map<String, Long> extendedRanges = new HashMap<>(varRanges);
                extendedRanges.put(loop.indexName(), bound);

                // Rewrite body with extended context
                IndexSimplificationRewriter bodyRewriter =
                        new IndexSimplificationRewriter(extendedRanges);
                LIRNode newBody = loop.body().accept(bodyRewriter);

                if (newBody == loop.body()) {
                    return loop;
                }
                return new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
            }

            // Non-constant bound, just rewrite body with current context
            LIRNode newBody = loop.body().accept(this);
            if (newBody == loop.body()) {
                return loop;
            }
            return new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
        }

        @Override
        public LIRNode visitStructuredFor(StructuredFor loop) {
            long upper = extractConstantBound(loop.upperBound());
            long lower = extractConstantBound(loop.lowerBound());
            long step = extractConstantBound(loop.step());

            if (upper > 0 && lower >= 0 && step > 0) {
                Map<String, Long> extendedRanges = new HashMap<>(varRanges);
                extendedRanges.put(loop.indexName(), upper - lower);

                IndexSimplificationRewriter bodyRewriter =
                        new IndexSimplificationRewriter(extendedRanges);
                LIRNode newBody = loop.body().accept(bodyRewriter);

                if (newBody == loop.body()) {
                    return loop;
                }
                return new StructuredFor(
                        loop.indexName(),
                        loop.lowerBound(),
                        loop.upperBound(),
                        loop.step(),
                        loop.iterArgs(),
                        newBody);
            }

            LIRNode newBody = loop.body().accept(this);
            if (newBody == loop.body()) {
                return loop;
            }
            return new StructuredFor(
                    loop.indexName(),
                    loop.lowerBound(),
                    loop.upperBound(),
                    loop.step(),
                    loop.iterArgs(),
                    newBody);
        }

        @Override
        public LIRNode visitTiledLoop(TiledLoop tiled) {
            // Extract the total bound if it's a constant
            long bound = extractConstantBound(tiled.totalBound());

            if (bound > 0) {
                // Create extended context with outer and inner variable ranges
                // Note: For tiled loops, both indices range over portions of the total
                // This is a simplification - in reality the ranges are more complex
                Map<String, Long> extendedRanges = new HashMap<>(varRanges);
                // We don't add exact ranges for tiled indices as they're complex
                // But we pass the context down for nested structures

                // Rewrite body with extended context
                IndexSimplificationRewriter bodyRewriter =
                        new IndexSimplificationRewriter(extendedRanges);
                LIRNode newBody = tiled.body().accept(bodyRewriter);

                if (newBody == tiled.body()) {
                    return tiled;
                }
                return new TiledLoop(
                        tiled.outerName(),
                        tiled.innerName(),
                        tiled.totalBound(),
                        tiled.tileSize(),
                        newBody);
            }

            // Non-constant bound, just rewrite body with current context
            LIRNode newBody = tiled.body().accept(this);
            if (newBody == tiled.body()) {
                return tiled;
            }
            return new TiledLoop(
                    tiled.outerName(),
                    tiled.innerName(),
                    tiled.totalBound(),
                    tiled.tileSize(),
                    newBody);
        }

        @Override
        public LIRNode visitLoopNest(LoopNest nest) {
            // Extract ranges from all loops in the nest
            Map<String, Long> extendedRanges = new HashMap<>(varRanges);
            boolean hasConstantBounds = false;

            for (Loop loop : nest.loops()) {
                long bound = extractConstantBound(loop.bound());
                if (bound > 0) {
                    extendedRanges.put(loop.indexName(), bound);
                    hasConstantBounds = true;
                }
            }

            if (hasConstantBounds) {
                // Rewrite body with extended context
                IndexSimplificationRewriter bodyRewriter =
                        new IndexSimplificationRewriter(extendedRanges);
                LIRNode newBody = nest.body().accept(bodyRewriter);

                if (newBody == nest.body()) {
                    return nest;
                }
                return new LoopNest(nest.loops(), newBody);
            }

            // No constant bounds, just rewrite body with current context
            LIRNode newBody = nest.body().accept(this);
            if (newBody == nest.body()) {
                return nest;
            }
            return new LoopNest(nest.loops(), newBody);
        }

        @Override
        public LIRNode visitBlock(Block block) {
            List<LIRNode> newStatements = new java.util.ArrayList<>();
            boolean changed = false;

            for (LIRNode stmt : block.statements()) {
                LIRNode newStmt = stmt.accept(this);
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

        @Override
        public LIRNode visitIndexBinary(IndexBinary node) {
            // First visit children recursively
            IndexExpr newLeft = (IndexExpr) node.left().accept(this);
            IndexExpr newRight = (IndexExpr) node.right().accept(this);

            // Create potentially updated node
            IndexBinary updated;
            if (newLeft == node.left() && newRight == node.right()) {
                updated = node;
            } else {
                updated = new IndexBinary(node.op(), newLeft, newRight);
            }

            // First apply standard simplifications
            IndexExpr simplified = updated.simplify();

            // Then try range-based simplifications if it's still a binary op
            if (simplified instanceof IndexBinary bin) {
                simplified = simplifyWithRanges(bin, varRanges);
            }

            return simplified;
        }

        @Override
        public LIRNode visitScalarLet(ScalarLet node) {
            // Visit the value expression with current range context
            ScalarExpr newValue = (ScalarExpr) node.value().accept(this);
            if (newValue == node.value()) {
                return node;
            }
            return new ScalarLet(node.name(), newValue);
        }

        @Override
        public LIRNode visitScalarFromIndex(ScalarFromIndex node) {
            // Simplify the index expression
            IndexExpr newIndex = (IndexExpr) node.index().accept(this);
            if (newIndex == node.index()) {
                return node;
            }
            return new ScalarFromIndex(newIndex);
        }

        @Override
        public LIRNode visitScalarLoad(ScalarLoad node) {
            // Simplify the offset expression
            IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
            if (newOffset == node.offset()) {
                return node;
            }
            return new ScalarLoad(node.buffer(), newOffset);
        }

        @Override
        public LIRNode visitStore(Store node) {
            // Simplify the offset expression
            IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
            ScalarExpr newValue = (ScalarExpr) node.value().accept(this);

            if (newOffset == node.offset() && newValue == node.value()) {
                return node;
            }
            return new Store(node.buffer(), newOffset, newValue);
        }
    }
}

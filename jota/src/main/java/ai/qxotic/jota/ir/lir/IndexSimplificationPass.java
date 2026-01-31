package ai.qxotic.jota.ir.lir;

/**
 * Pass that simplifies index expressions using algebraic rules. Leverages the existing
 * simplification logic in {@link IndexBinary} and {@link IndexExpr#simplify()}.
 *
 * <p>Simplifications include:
 *
 * <ul>
 *   <li>Identity elimination: x + 0 → x, x * 1 → x, x / 1 → x
 *   <li>Zero propagation: x * 0 → 0, 0 / x → 0
 *   <li>Constant folding: 3 + 4 → 7
 *   <li>Strength reduction: x * 2^n → x << n, x / 2^n → x >> n
 *   <li>Modulo optimization: x % 2^n → x & (2^n - 1)
 * </ul>
 */
public final class IndexSimplificationPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        return new IndexSimplificationRewriter().rewrite(graph);
    }

    @Override
    public String name() {
        return "IndexSimplification";
    }

    private static final class IndexSimplificationRewriter extends LIRRewriter {

        @Override
        public LIRNode visitIndexBinary(IndexBinary node) {
            // First visit children
            LIRNode visited = super.visitIndexBinary(node);
            if (visited instanceof IndexExpr expr) {
                // Then simplify the result
                return expr.simplify();
            }
            return visited;
        }
    }
}

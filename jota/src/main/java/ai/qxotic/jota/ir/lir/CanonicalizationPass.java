package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.Set;

/**
 * Pass that rewrites expressions into canonical form. For commutative operations, operands are
 * reordered according to a consistent ordering. This improves the effectiveness of CSE by ensuring
 * that equivalent expressions like {@code a + b} and {@code b + a} have the same representation.
 *
 * <p>Canonical ordering rules:
 *
 * <ol>
 *   <li>Constants come before variables
 *   <li>Variables come before complex expressions
 *   <li>Among constants: ordered by value
 *   <li>Among variables: ordered by name
 *   <li>Among complex expressions: ordered by operator, then recursively by operands
 * </ol>
 */
public final class CanonicalizationPass implements LIRPass {

    /** Commutative index binary operations. */
    private static final Set<IndexBinary.IndexBinaryOp> COMMUTATIVE_INDEX_OPS =
            Set.of(
                    IndexBinary.IndexBinaryOp.ADD,
                    IndexBinary.IndexBinaryOp.MULTIPLY,
                    IndexBinary.IndexBinaryOp.BITWISE_AND);

    /** Commutative scalar binary operations. */
    private static final Set<BinaryOperator> COMMUTATIVE_SCALAR_OPS =
            Set.of(
                    BinaryOperator.ADD,
                    BinaryOperator.MULTIPLY,
                    BinaryOperator.MIN,
                    BinaryOperator.MAX,
                    BinaryOperator.LOGICAL_AND,
                    BinaryOperator.LOGICAL_OR,
                    BinaryOperator.LOGICAL_XOR,
                    BinaryOperator.BITWISE_AND,
                    BinaryOperator.BITWISE_OR,
                    BinaryOperator.BITWISE_XOR,
                    BinaryOperator.EQUAL);

    @Override
    public LIRGraph run(LIRGraph graph) {
        return new CanonicalizationRewriter().rewrite(graph);
    }

    @Override
    public String name() {
        return "Canonicalization";
    }

    /**
     * Compares two index expressions for canonical ordering.
     *
     * <p>Order: IndexConst < IndexVar < IndexBinary
     */
    private static int compareIndex(IndexExpr a, IndexExpr b) {
        int typeA = indexTypeOrder(a);
        int typeB = indexTypeOrder(b);
        if (typeA != typeB) {
            return Integer.compare(typeA, typeB);
        }
        // Same type, compare within type
        return switch (a) {
            case IndexConst ca -> Long.compare(ca.value(), ((IndexConst) b).value());
            case IndexVar va -> va.name().compareTo(((IndexVar) b).name());
            case IndexBinary ba -> {
                IndexBinary bb = (IndexBinary) b;
                int opCmp = ba.op().compareTo(bb.op());
                if (opCmp != 0) yield opCmp;
                int leftCmp = compareIndex(ba.left(), bb.left());
                if (leftCmp != 0) yield leftCmp;
                yield compareIndex(ba.right(), bb.right());
            }
        };
    }

    private static int indexTypeOrder(IndexExpr e) {
        return switch (e) {
            case IndexConst __ -> 0;
            case IndexVar __ -> 1;
            case IndexBinary __ -> 2;
        };
    }

    /**
     * Compares two scalar expressions for canonical ordering.
     *
     * <p>Order: ScalarLiteral < ScalarInput < ScalarRef < ScalarLoad < ScalarFromIndex < ScalarCast
     * < ScalarUnary < ScalarTernary < ScalarBinary
     */
    private static int compareScalar(ScalarExpr a, ScalarExpr b) {
        int typeA = scalarTypeOrder(a);
        int typeB = scalarTypeOrder(b);
        if (typeA != typeB) {
            return Integer.compare(typeA, typeB);
        }
        // Same type, compare within type
        return switch (a) {
            case ScalarLiteral ca -> {
                ScalarLiteral cb = (ScalarLiteral) b;
                int dtCmp = ca.dataType().name().compareTo(cb.dataType().name());
                if (dtCmp != 0) yield dtCmp;
                yield Long.compare(ca.rawBits(), cb.rawBits());
            }
            case ScalarLoad la -> {
                ScalarLoad lb = (ScalarLoad) b;
                int bufCmp = Integer.compare(la.buffer().id(), lb.buffer().id());
                if (bufCmp != 0) yield bufCmp;
                yield compareIndex(la.offset(), lb.offset());
            }
            case ScalarInput ia -> Integer.compare(ia.id(), ((ScalarInput) b).id());
            case ScalarFromIndex fa -> compareIndex(fa.index(), ((ScalarFromIndex) b).index());
            case ScalarCast ca -> {
                ScalarCast cb = (ScalarCast) b;
                int dtCmp = ca.targetType().name().compareTo(cb.targetType().name());
                if (dtCmp != 0) yield dtCmp;
                yield compareScalar(ca.input(), cb.input());
            }
            case ScalarUnary ua -> {
                ScalarUnary ub = (ScalarUnary) b;
                int opCmp = ua.op().compareTo(ub.op());
                if (opCmp != 0) yield opCmp;
                yield compareScalar(ua.input(), ub.input());
            }
            case ScalarTernary ta -> {
                ScalarTernary tb = (ScalarTernary) b;
                int condCmp = compareScalar(ta.condition(), tb.condition());
                if (condCmp != 0) yield condCmp;
                int trueCmp = compareScalar(ta.trueValue(), tb.trueValue());
                if (trueCmp != 0) yield trueCmp;
                yield compareScalar(ta.falseValue(), tb.falseValue());
            }
            case ScalarBinary ba -> {
                ScalarBinary bb = (ScalarBinary) b;
                int opCmp = ba.op().compareTo(bb.op());
                if (opCmp != 0) yield opCmp;
                int leftCmp = compareScalar(ba.left(), bb.left());
                if (leftCmp != 0) yield leftCmp;
                yield compareScalar(ba.right(), bb.right());
            }
            case ScalarRef ra -> ra.name().compareTo(((ScalarRef) b).name());
        };
    }

    private static int scalarTypeOrder(ScalarExpr e) {
        return switch (e) {
            case ScalarLiteral __ -> 0;
            case ScalarInput __ -> 1;
            case ScalarRef __ -> 2;
            case ScalarLoad __ -> 3;
            case ScalarFromIndex __ -> 4;
            case ScalarCast __ -> 5;
            case ScalarUnary __ -> 6;
            case ScalarTernary __ -> 7;
            case ScalarBinary __ -> 8;
        };
    }

    private static final class CanonicalizationRewriter extends LIRRewriter {

        @Override
        public LIRNode visitIndexBinary(IndexBinary node) {
            // First visit children
            IndexExpr newLeft = (IndexExpr) node.left().accept(this);
            IndexExpr newRight = (IndexExpr) node.right().accept(this);

            // Check if commutative and needs reordering
            if (COMMUTATIVE_INDEX_OPS.contains(node.op())) {
                if (compareIndex(newLeft, newRight) > 0) {
                    // Swap operands
                    IndexExpr temp = newLeft;
                    newLeft = newRight;
                    newRight = temp;
                }
            }

            if (newLeft == node.left() && newRight == node.right()) {
                return node;
            }
            return new IndexBinary(node.op(), newLeft, newRight);
        }

        @Override
        public LIRNode visitScalarBinary(ScalarBinary node) {
            // First visit children
            ScalarExpr newLeft = (ScalarExpr) node.left().accept(this);
            ScalarExpr newRight = (ScalarExpr) node.right().accept(this);

            // Check if commutative and needs reordering
            if (COMMUTATIVE_SCALAR_OPS.contains(node.op())) {
                if (compareScalar(newLeft, newRight) > 0) {
                    // Swap operands
                    ScalarExpr temp = newLeft;
                    newLeft = newRight;
                    newRight = temp;
                }
            }

            if (newLeft == node.left() && newRight == node.right()) {
                return node;
            }
            return new ScalarBinary(node.op(), newLeft, newRight);
        }
    }
}

package ai.qxotic.jota.ir.lir;

import java.util.HashMap;
import java.util.Map;

/**
 * Pass that eliminates common subexpressions. When the same expression appears multiple times, this
 * pass ensures they all reference the same instance.
 *
 * <p>This pass uses Java's built-in record equality for structural comparison. It performs a
 * post-order traversal to canonicalize children before parents, ensuring that equivalent
 * expressions are recognized even when constructed independently.
 *
 * <p>CSE is beneficial for:
 *
 * <ul>
 *   <li>Index expressions: offset calculations often share common terms
 *   <li>Scalar expressions: repeated computations within loop bodies
 * </ul>
 */
public final class CommonSubexpressionElimination implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        return new CSERewriter().rewrite(graph);
    }

    @Override
    public String name() {
        return "CommonSubexpressionElimination";
    }

    private static final class CSERewriter extends LIRRewriter {
        private final Map<LIRNode, LIRNode> canonical = new HashMap<>();

        private <T extends LIRNode> T canonicalize(T node) {
            @SuppressWarnings("unchecked")
            T existing = (T) canonical.putIfAbsent(node, node);
            return existing != null ? existing : node;
        }

        // Index expressions

        @Override
        public LIRNode visitIndexVar(IndexVar node) {
            return canonicalize(node);
        }

        @Override
        public LIRNode visitIndexConst(IndexConst node) {
            return canonicalize(node);
        }

        @Override
        public LIRNode visitIndexBinary(IndexBinary node) {
            IndexExpr newLeft = (IndexExpr) node.left().accept(this);
            IndexExpr newRight = (IndexExpr) node.right().accept(this);
            IndexBinary result;
            if (newLeft == node.left() && newRight == node.right()) {
                result = node;
            } else {
                result = new IndexBinary(node.op(), newLeft, newRight);
            }
            return canonicalize(result);
        }

        // Scalar expressions

        @Override
        public LIRNode visitScalarLiteral(ScalarLiteral node) {
            return canonicalize(node);
        }

        @Override
        public LIRNode visitScalarUnary(ScalarUnary node) {
            ScalarExpr newInput = (ScalarExpr) node.input().accept(this);
            ScalarUnary result;
            if (newInput == node.input()) {
                result = node;
            } else {
                result = new ScalarUnary(node.op(), newInput);
            }
            return canonicalize(result);
        }

        @Override
        public LIRNode visitScalarBinary(ScalarBinary node) {
            ScalarExpr newLeft = (ScalarExpr) node.left().accept(this);
            ScalarExpr newRight = (ScalarExpr) node.right().accept(this);
            ScalarBinary result;
            if (newLeft == node.left() && newRight == node.right()) {
                result = node;
            } else {
                result = new ScalarBinary(node.op(), newLeft, newRight);
            }
            return canonicalize(result);
        }

        @Override
        public LIRNode visitScalarTernary(ScalarTernary node) {
            ScalarExpr newCondition = (ScalarExpr) node.condition().accept(this);
            ScalarExpr newTrueValue = (ScalarExpr) node.trueValue().accept(this);
            ScalarExpr newFalseValue = (ScalarExpr) node.falseValue().accept(this);
            ScalarTernary result;
            if (newCondition == node.condition()
                    && newTrueValue == node.trueValue()
                    && newFalseValue == node.falseValue()) {
                result = node;
            } else {
                result = new ScalarTernary(newCondition, newTrueValue, newFalseValue);
            }
            return canonicalize(result);
        }

        @Override
        public LIRNode visitScalarCast(ScalarCast node) {
            ScalarExpr newInput = (ScalarExpr) node.input().accept(this);
            ScalarCast result;
            if (newInput == node.input()) {
                result = node;
            } else {
                result = new ScalarCast(newInput, node.targetType());
            }
            return canonicalize(result);
        }

        @Override
        public LIRNode visitScalarLoad(ScalarLoad node) {
            IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
            ScalarLoad result;
            if (newOffset == node.offset()) {
                result = node;
            } else {
                result = new ScalarLoad(node.buffer(), newOffset);
            }
            return canonicalize(result);
        }

        @Override
        public LIRNode visitScalarInput(ScalarInput node) {
            return canonicalize(node);
        }

        @Override
        public LIRNode visitScalarFromIndex(ScalarFromIndex node) {
            IndexExpr newIndex = (IndexExpr) node.index().accept(this);
            ScalarFromIndex result;
            if (newIndex == node.index()) {
                result = node;
            } else {
                result = new ScalarFromIndex(newIndex);
            }
            return canonicalize(result);
        }

        // Memory access - BufferRef should be canonicalized but Load/Store are statements

        @Override
        public LIRNode visitBufferRef(BufferRef node) {
            return canonicalize(node);
        }

        @Override
        public LIRNode visitLoad(Load node) {
            IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
            Load result;
            if (newOffset == node.offset()) {
                result = node;
            } else {
                result = new Load(node.buffer(), newOffset);
            }
            return canonicalize(result);
        }

        // Statements are not canonicalized but their children are

        @Override
        public LIRNode visitStore(Store node) {
            IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
            ScalarExpr newValue = (ScalarExpr) node.value().accept(this);
            if (newOffset == node.offset() && newValue == node.value()) {
                return node;
            }
            return new Store(node.buffer(), newOffset, newValue);
        }

        // Control flow - not canonicalized, but children are processed

        @Override
        public LIRNode visitLoop(Loop node) {
            IndexExpr newBound = (IndexExpr) node.bound().accept(this);
            LIRNode newBody = node.body().accept(this);
            if (newBound == node.bound() && newBody == node.body()) {
                return node;
            }
            return new Loop(node.indexName(), newBound, node.isParallel(), newBody);
        }

        @Override
        public LIRNode visitStructuredFor(StructuredFor node) {
            IndexExpr newLower = (IndexExpr) node.lowerBound().accept(this);
            IndexExpr newUpper = (IndexExpr) node.upperBound().accept(this);
            IndexExpr newStep = (IndexExpr) node.step().accept(this);

            java.util.List<LoopIterArg> newIterArgs =
                    new java.util.ArrayList<>(node.iterArgs().size());
            boolean iterChanged = false;
            for (LoopIterArg arg : node.iterArgs()) {
                ScalarExpr newInit = (ScalarExpr) arg.init().accept(this);
                if (newInit != arg.init()) {
                    iterChanged = true;
                    newIterArgs.add(new LoopIterArg(arg.name(), arg.dataType(), newInit));
                } else {
                    newIterArgs.add(arg);
                }
            }

            LIRNode newBody = node.body().accept(this);
            if (newLower == node.lowerBound()
                    && newUpper == node.upperBound()
                    && newStep == node.step()
                    && !iterChanged
                    && newBody == node.body()) {
                return node;
            }
            return new StructuredFor(
                    node.indexName(), newLower, newUpper, newStep, newIterArgs, newBody);
        }

        @Override
        public LIRNode visitTiledLoop(TiledLoop node) {
            IndexExpr newTotalBound = (IndexExpr) node.totalBound().accept(this);
            LIRNode newBody = node.body().accept(this);
            if (newTotalBound == node.totalBound() && newBody == node.body()) {
                return node;
            }
            return new TiledLoop(
                    node.outerName(), node.innerName(), newTotalBound, node.tileSize(), newBody);
        }

        @Override
        public LIRNode visitYield(Yield node) {
            java.util.List<ScalarExpr> newValues = new java.util.ArrayList<>(node.values().size());
            boolean changed = false;
            for (ScalarExpr value : node.values()) {
                ScalarExpr newValue = (ScalarExpr) value.accept(this);
                newValues.add(newValue);
                if (newValue != value) {
                    changed = true;
                }
            }
            if (!changed) {
                return node;
            }
            return new Yield(newValues);
        }
    }
}

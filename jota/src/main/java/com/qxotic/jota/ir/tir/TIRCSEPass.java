package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * TIR pass that eliminates common subexpressions (CSE).
 *
 * <p>This pass identifies and shares identical subexpressions within the graph, reducing redundant
 * computation and graph size. It uses structural equality (based on operation type and inputs) to
 * identify common subexpressions.
 *
 * <p>For example:
 *
 * <ul>
 *   <li>{@code a = x * y; b = x * y} → {@code a = x * y; b = a} (share the multiply)
 *   <li>{@code a = x + y; b = y + x} → NOT shared (order matters)
 * </ul>
 */
public final class TIRCSEPass implements TIRPass {

    @Override
    public TIRGraph run(TIRGraph graph) {
        CSERewriter rewriter = new CSERewriter();
        return rewriter.rewrite(graph);
    }

    /**
     * Key for identifying structurally equivalent expressions. Two expressions are equivalent if
     * they have the same operation, same data type, same shape, and equivalent inputs.
     */
    private record ExprKey(
            Class<? extends TIRNode> nodeClass,
            Object op, // UnaryOperator, BinaryOperator, TernaryOperator, etc.
            DataType dataType,
            Shape shape,
            TIRNode[] inputs) {

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof ExprKey other)) return false;
            if (nodeClass != other.nodeClass) return false;
            if (!Objects.equals(op, other.op)) return false;
            if (dataType != other.dataType) return false;
            if (!shape.equals(other.shape)) return false;
            if (inputs.length != other.inputs.length) return false;
            // Use identity equality for inputs - they must be the same node instance
            for (int i = 0; i < inputs.length; i++) {
                if (inputs[i] != other.inputs[i]) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            int result = nodeClass.hashCode();
            result = 31 * result + Objects.hashCode(op);
            result = 31 * result + dataType.hashCode();
            result = 31 * result + shape.hashCode();
            // Include identity hash codes of inputs
            for (TIRNode input : inputs) {
                result = 31 * result + System.identityHashCode(input);
            }
            return result;
        }
    }

    private static final class CSERewriter extends TIRRewriter {
        private final Map<ExprKey, TIRNode> exprCache = new HashMap<>();
        private final Map<TIRNode, TIRNode> replacementMap = new HashMap<>();

        @Override
        public TIRGraph rewrite(TIRGraph graph) {
            // First pass: collect all expressions and identify duplicates
            for (TIRNode input : graph.inputs()) {
                collectExpressions(input);
            }
            for (TIRNode output : graph.outputs()) {
                collectExpressions(output);
            }

            // Second pass: use parent rewriter to rebuild graph with replacements
            // The visit methods will use getReplacement() to substitute duplicate nodes
            return super.rewrite(graph);
        }

        private void collectExpressions(TIRNode node) {
            if (replacementMap.containsKey(node)) {
                return; // Already processed
            }

            // First process children
            switch (node) {
                case UnaryOp op -> collectExpressions(op.input());
                case BinaryOp op -> {
                    collectExpressions(op.left());
                    collectExpressions(op.right());
                }
                case TernaryOp op -> {
                    collectExpressions(op.cond());
                    collectExpressions(op.trueExpr());
                    collectExpressions(op.falseExpr());
                }
                case CastOp op -> collectExpressions(op.input());
                case ReductionOp op -> collectExpressions(op.input());
                case ViewTransform vt -> collectExpressions(vt.input());
                case Contiguous contig -> collectExpressions(contig.input());
                default -> {
                    // Leaf nodes (TensorInput, ScalarInput, ScalarConstant, IotaConstant)
                }
            }

            // Now try to find or create a replacement
            ExprKey key = createKey(node);
            if (key != null) {
                TIRNode existing = exprCache.get(key);
                if (existing != null) {
                    // Found a duplicate - map to the existing node
                    replacementMap.put(node, existing);
                } else {
                    // First time seeing this expression
                    exprCache.put(key, node);
                    replacementMap.put(node, node);
                }
            } else {
                // Nodes that don't participate in CSE (inputs, constants)
                replacementMap.put(node, node);
            }
        }

        private ExprKey createKey(TIRNode node) {
            return switch (node) {
                case UnaryOp op -> {
                    TIRNode input = getReplacement(op.input());
                    yield new ExprKey(
                            UnaryOp.class,
                            op.op(),
                            op.dataType(),
                            op.shape(),
                            new TIRNode[] {input});
                }
                case BinaryOp op -> {
                    TIRNode left = getReplacement(op.left());
                    TIRNode right = getReplacement(op.right());
                    yield new ExprKey(
                            BinaryOp.class,
                            op.op(),
                            op.dataType(),
                            op.shape(),
                            new TIRNode[] {left, right});
                }
                case TernaryOp op -> {
                    TIRNode cond = getReplacement(op.cond());
                    TIRNode trueExpr = getReplacement(op.trueExpr());
                    TIRNode falseExpr = getReplacement(op.falseExpr());
                    yield new ExprKey(
                            TernaryOp.class,
                            op.op(),
                            op.dataType(),
                            op.shape(),
                            new TIRNode[] {cond, trueExpr, falseExpr});
                }
                case CastOp op -> {
                    TIRNode input = getReplacement(op.input());
                    yield new ExprKey(
                            CastOp.class,
                            op.targetDataType(),
                            op.dataType(),
                            op.shape(),
                            new TIRNode[] {input});
                }
                case ReductionOp op -> {
                    TIRNode input = getReplacement(op.input());
                    yield new ExprKey(
                            ReductionOp.class,
                            new ReductionKey(
                                    op.op(), op.axes(), op.keepDims(), op.accumulatorType()),
                            op.dataType(),
                            op.shape(),
                            new TIRNode[] {input});
                }
                // Don't CSE ViewTransform and Contiguous - they have layout semantics
                default -> null;
            };
        }

        private TIRNode getReplacement(TIRNode node) {
            TIRNode replacement = replacementMap.get(node);
            return replacement != null ? replacement : node;
        }

        @Override
        public TIRNode visitUnaryOp(UnaryOp node) {
            TIRNode newInput = node.input().accept(this);
            TIRNode result = getReplacement(node);
            if (result == node) {
                // Not replaced - need to create new node with rewritten input
                if (newInput == node.input()) {
                    return node;
                }
                return new UnaryOp(node.op(), newInput, node.shape());
            }
            return result;
        }

        @Override
        public TIRNode visitBinaryOp(BinaryOp node) {
            TIRNode newLeft = node.left().accept(this);
            TIRNode newRight = node.right().accept(this);
            TIRNode result = getReplacement(node);
            if (result == node) {
                // Not replaced - need to create new node with rewritten inputs
                if (newLeft == node.left() && newRight == node.right()) {
                    return node;
                }
                return new BinaryOp(node.op(), newLeft, newRight, node.shape());
            }
            return result;
        }

        @Override
        public TIRNode visitTernaryOp(TernaryOp node) {
            TIRNode newCond = node.cond().accept(this);
            TIRNode newTrue = node.trueExpr().accept(this);
            TIRNode newFalse = node.falseExpr().accept(this);
            TIRNode result = getReplacement(node);
            if (result == node) {
                // Not replaced
                if (newCond == node.cond()
                        && newTrue == node.trueExpr()
                        && newFalse == node.falseExpr()) {
                    return node;
                }
                return new TernaryOp(node.op(), newCond, newTrue, newFalse, node.shape());
            }
            return result;
        }

        @Override
        public TIRNode visitCastOp(CastOp node) {
            TIRNode newInput = node.input().accept(this);
            TIRNode result = getReplacement(node);
            if (result == node) {
                if (newInput == node.input()) {
                    return node;
                }
                return new CastOp(newInput, node.targetDataType(), node.shape());
            }
            return result;
        }

        @Override
        public TIRNode visitReductionOp(ReductionOp node) {
            TIRNode newInput = node.input().accept(this);
            TIRNode result = getReplacement(node);
            if (result == node) {
                if (newInput == node.input()) {
                    return node;
                }
                return new ReductionOp(
                        node.op(),
                        newInput,
                        node.axes(),
                        node.keepDims(),
                        node.accumulatorType(),
                        node.shape());
            }
            return result;
        }

        @Override
        public TIRNode visitViewTransform(ViewTransform node) {
            TIRNode newInput = node.input().accept(this);
            if (newInput == node.input()) {
                return node;
            }
            return new ViewTransform(
                    newInput, node.kind(), node.layout(), node.needsLazyIndexing());
        }

        @Override
        public TIRNode visitContiguous(Contiguous node) {
            TIRNode newInput = node.input().accept(this);
            if (newInput == node.input()) {
                return node;
            }
            return new Contiguous(newInput, node.shape());
        }
    }

    /** Key for reduction operations that includes all reduction parameters. */
    private record ReductionKey(
            ReductionOperator op, int[] axes, boolean keepDims, DataType accumulatorType) {

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof ReductionKey other)) return false;
            if (op != other.op) return false;
            if (keepDims != other.keepDims) return false;
            if (accumulatorType != other.accumulatorType) return false;
            if (axes.length != other.axes.length) return false;
            for (int i = 0; i < axes.length; i++) {
                if (axes[i] != other.axes[i]) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            int result = op.hashCode();
            result = 31 * result + java.util.Arrays.hashCode(axes);
            result = 31 * result + Boolean.hashCode(keepDims);
            result = 31 * result + accumulatorType.hashCode();
            return result;
        }
    }
}

package com.qxotic.jota.ir.tir;

import java.util.ArrayList;
import java.util.List;

/**
 * Base visitor for rewriting TIR graphs. Recursively visits and rebuilds the tree, using structural
 * sharing when nodes are unchanged. Subclasses override specific visit methods to transform nodes.
 *
 * <p>The default implementation of each visit method:
 *
 * <ol>
 *   <li>Recursively visits all children
 *   <li>If all children are unchanged, returns the original node (structural sharing)
 *   <li>Otherwise, creates a new node with the transformed children
 * </ol>
 */
public class TIRRewriter implements TIRVisitor<TIRNode> {

    private final java.util.Map<TIRNode, TIRNode> rewriteCache = new java.util.IdentityHashMap<>();

    /**
     * Rewrites a TIRGraph by transforming its outputs.
     *
     * @param graph the graph to rewrite
     * @return the rewritten graph
     */
    public TIRGraph rewrite(TIRGraph graph) {
        List<TIRNode> newOutputs = new ArrayList<>(graph.outputs().size());
        boolean changed = false;
        for (TIRNode output : graph.outputs()) {
            TIRNode newOutput = rewriteChild(output);
            newOutputs.add(newOutput);
            if (newOutput != output) {
                changed = true;
            }
        }
        if (!changed) {
            return graph;
        }
        // Recompute inputs from the new outputs
        List<TIRNode> newInputs = computeInputsFromOutputs(newOutputs);
        return new TIRGraph(newInputs, newOutputs);
    }

    /** Computes the list of input nodes that are actually referenced by the given outputs. */
    private List<TIRNode> computeInputsFromOutputs(List<TIRNode> outputs) {
        java.util.Set<TIRNode> inputs =
                java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        java.util.Set<TIRNode> visited =
                java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        for (TIRNode output : outputs) {
            collectInputs(output, inputs, visited);
        }
        return new ArrayList<>(inputs);
    }

    /** Recursively collects all TensorInput nodes referenced by a node. */
    private void collectInputs(
            TIRNode node, java.util.Set<TIRNode> inputs, java.util.Set<TIRNode> visited) {
        if (!visited.add(node)) {
            return;
        }
        switch (node) {
            case TensorInput ti -> inputs.add(ti);
            case ScalarInput si -> inputs.add(si);
            case UnaryOp op -> collectInputs(op.input(), inputs, visited);
            case BinaryOp op -> {
                collectInputs(op.left(), inputs, visited);
                collectInputs(op.right(), inputs, visited);
            }
            case TernaryOp op -> {
                collectInputs(op.cond(), inputs, visited);
                collectInputs(op.trueExpr(), inputs, visited);
                collectInputs(op.falseExpr(), inputs, visited);
            }
            case CastOp op -> collectInputs(op.input(), inputs, visited);
            case ReductionOp op -> collectInputs(op.input(), inputs, visited);
            case GatherOp op -> {
                collectInputs(op.input(), inputs, visited);
                collectInputs(op.indices(), inputs, visited);
            }
            case ViewTransform vt -> collectInputs(vt.input(), inputs, visited);
            case Contiguous c -> collectInputs(c.input(), inputs, visited);
            case ScalarConstant sc -> {
                // Leaf node - no inputs
            }
            case IotaConstant ic -> {
                // Leaf node - no inputs
            }
            case RandomUniformOp ru -> {
                // Leaf node - no inputs
            }
        }
    }

    @Override
    public TIRNode visitTensorInput(TensorInput node) {
        return node;
    }

    public TIRNode visitScalarInput(ScalarInput node) {
        return node;
    }

    @Override
    public TIRNode visitScalarConstant(ScalarConstant node) {
        return node;
    }

    @Override
    public TIRNode visitIotaConstant(IotaConstant node) {
        return node;
    }

    @Override
    public TIRNode visitRandomUniformOp(RandomUniformOp node) {
        return node;
    }

    @Override
    public TIRNode visitUnaryOp(UnaryOp node) {
        TIRNode newInput = rewriteChild(node.input());
        if (newInput == node.input()) {
            return node;
        }
        return new UnaryOp(node.op(), newInput, node.shape());
    }

    @Override
    public TIRNode visitBinaryOp(BinaryOp node) {
        TIRNode newLeft = rewriteChild(node.left());
        TIRNode newRight = rewriteChild(node.right());
        if (newLeft == node.left() && newRight == node.right()) {
            return node;
        }
        return new BinaryOp(node.op(), newLeft, newRight, node.shape());
    }

    @Override
    public TIRNode visitTernaryOp(TernaryOp node) {
        TIRNode newCond = rewriteChild(node.cond());
        TIRNode newTrue = rewriteChild(node.trueExpr());
        TIRNode newFalse = rewriteChild(node.falseExpr());
        if (newCond == node.cond() && newTrue == node.trueExpr() && newFalse == node.falseExpr()) {
            return node;
        }
        return new TernaryOp(node.op(), newCond, newTrue, newFalse, node.shape());
    }

    @Override
    public TIRNode visitCastOp(CastOp node) {
        TIRNode newInput = rewriteChild(node.input());
        if (newInput == node.input()) {
            return node;
        }
        return new CastOp(newInput, node.targetDataType(), node.shape());
    }

    @Override
    public TIRNode visitReductionOp(ReductionOp node) {
        TIRNode newInput = rewriteChild(node.input());
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

    @Override
    public TIRNode visitGatherOp(GatherOp node) {
        TIRNode newInput = rewriteChild(node.input());
        TIRNode newIndices = rewriteChild(node.indices());
        if (newInput == node.input() && newIndices == node.indices()) {
            return node;
        }
        return new GatherOp(newInput, newIndices, node.axis(), node.shape());
    }

    @Override
    public TIRNode visitViewTransform(ViewTransform node) {
        TIRNode newInput = rewriteChild(node.input());
        if (newInput == node.input()) {
            return node;
        }
        return new ViewTransform(newInput, node.kind(), node.layout(), node.needsLazyIndexing());
    }

    @Override
    public TIRNode visitContiguous(Contiguous node) {
        TIRNode newInput = rewriteChild(node.input());
        if (newInput == node.input()) {
            return node;
        }
        return new Contiguous(newInput, node.shape());
    }

    protected TIRNode rewriteChild(TIRNode node) {
        TIRNode cached = rewriteCache.get(node);
        if (cached != null) {
            return cached;
        }
        TIRNode result = node.accept(this);
        rewriteCache.put(node, result);
        return result;
    }
}

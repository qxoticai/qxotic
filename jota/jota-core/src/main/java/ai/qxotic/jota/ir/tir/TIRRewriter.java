package ai.qxotic.jota.ir.tir;

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
            TIRNode newOutput = output.accept(this);
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
        java.util.Set<TIRNode> inputs = new java.util.HashSet<>();
        for (TIRNode output : outputs) {
            collectInputs(output, inputs);
        }
        return new ArrayList<>(inputs);
    }

    /** Recursively collects all TensorInput nodes referenced by a node. */
    private void collectInputs(TIRNode node, java.util.Set<TIRNode> inputs) {
        switch (node) {
            case TensorInput ti -> inputs.add(ti);
            case ScalarInput si -> inputs.add(si);
            case UnaryOp op -> collectInputs(op.input(), inputs);
            case BinaryOp op -> {
                collectInputs(op.left(), inputs);
                collectInputs(op.right(), inputs);
            }
            case TernaryOp op -> {
                collectInputs(op.cond(), inputs);
                collectInputs(op.trueExpr(), inputs);
                collectInputs(op.falseExpr(), inputs);
            }
            case CastOp op -> collectInputs(op.input(), inputs);
            case ReductionOp op -> collectInputs(op.input(), inputs);
            case GatherOp op -> {
                collectInputs(op.input(), inputs);
                collectInputs(op.indices(), inputs);
            }
            case ViewTransform vt -> collectInputs(vt.input(), inputs);
            case Contiguous c -> collectInputs(c.input(), inputs);
            case ScalarConstant sc -> {
                // Leaf node - no inputs
            }
            case IotaConstant ic -> {
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
    public TIRNode visitUnaryOp(UnaryOp node) {
        TIRNode newInput = node.input().accept(this);
        if (newInput == node.input()) {
            return node;
        }
        return new UnaryOp(node.op(), newInput, node.shape());
    }

    @Override
    public TIRNode visitBinaryOp(BinaryOp node) {
        TIRNode newLeft = node.left().accept(this);
        TIRNode newRight = node.right().accept(this);
        if (newLeft == node.left() && newRight == node.right()) {
            return node;
        }
        return new BinaryOp(node.op(), newLeft, newRight, node.shape());
    }

    @Override
    public TIRNode visitTernaryOp(TernaryOp node) {
        TIRNode newCond = node.cond().accept(this);
        TIRNode newTrue = node.trueExpr().accept(this);
        TIRNode newFalse = node.falseExpr().accept(this);
        if (newCond == node.cond() && newTrue == node.trueExpr() && newFalse == node.falseExpr()) {
            return node;
        }
        return new TernaryOp(node.op(), newCond, newTrue, newFalse, node.shape());
    }

    @Override
    public TIRNode visitCastOp(CastOp node) {
        TIRNode newInput = node.input().accept(this);
        if (newInput == node.input()) {
            return node;
        }
        return new CastOp(newInput, node.targetDataType(), node.shape());
    }

    @Override
    public TIRNode visitReductionOp(ReductionOp node) {
        TIRNode newInput = node.input().accept(this);
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
        TIRNode newInput = node.input().accept(this);
        TIRNode newIndices = node.indices().accept(this);
        if (newInput == node.input() && newIndices == node.indices()) {
            return node;
        }
        return new GatherOp(newInput, newIndices, node.axis(), node.shape());
    }

    @Override
    public TIRNode visitViewTransform(ViewTransform node) {
        TIRNode newInput = node.input().accept(this);
        if (newInput == node.input()) {
            return node;
        }
        return new ViewTransform(newInput, node.kind(), node.layout(), node.needsLazyIndexing());
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

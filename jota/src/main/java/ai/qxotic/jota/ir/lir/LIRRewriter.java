package ai.qxotic.jota.ir.lir;

import java.util.ArrayList;
import java.util.List;

/**
 * Base visitor for rewriting LIR graphs. Recursively visits and rebuilds the tree, using structural
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
public class LIRRewriter implements LIRVisitor<LIRNode> {

    /**
     * Rewrites an LIRGraph by transforming its body.
     *
     * @param graph the graph to rewrite
     * @return the rewritten graph
     */
    public LIRGraph rewrite(LIRGraph graph) {
        LIRNode newBody = graph.body().accept(this);
        if (newBody == graph.body()) {
            return graph;
        }
        return new LIRGraph(graph.inputs(), graph.outputs(), newBody);
    }

    // Index expressions

    @Override
    public LIRNode visitIndexVar(IndexVar node) {
        return node;
    }

    @Override
    public LIRNode visitIndexConst(IndexConst node) {
        return node;
    }

    @Override
    public LIRNode visitIndexBinary(IndexBinary node) {
        IndexExpr newLeft = (IndexExpr) node.left().accept(this);
        IndexExpr newRight = (IndexExpr) node.right().accept(this);
        if (newLeft == node.left() && newRight == node.right()) {
            return node;
        }
        return new IndexBinary(node.op(), newLeft, newRight);
    }

    // Scalar expressions

    @Override
    public LIRNode visitScalarLiteral(ScalarLiteral node) {
        return node;
    }

    @Override
    public LIRNode visitScalarUnary(ScalarUnary node) {
        ScalarExpr newInput = (ScalarExpr) node.input().accept(this);
        if (newInput == node.input()) {
            return node;
        }
        return new ScalarUnary(node.op(), newInput);
    }

    @Override
    public LIRNode visitScalarBinary(ScalarBinary node) {
        ScalarExpr newLeft = (ScalarExpr) node.left().accept(this);
        ScalarExpr newRight = (ScalarExpr) node.right().accept(this);
        if (newLeft == node.left() && newRight == node.right()) {
            return node;
        }
        return new ScalarBinary(node.op(), newLeft, newRight);
    }

    @Override
    public LIRNode visitScalarTernary(ScalarTernary node) {
        ScalarExpr newCondition = (ScalarExpr) node.condition().accept(this);
        ScalarExpr newTrueValue = (ScalarExpr) node.trueValue().accept(this);
        ScalarExpr newFalseValue = (ScalarExpr) node.falseValue().accept(this);
        if (newCondition == node.condition()
                && newTrueValue == node.trueValue()
                && newFalseValue == node.falseValue()) {
            return node;
        }
        return new ScalarTernary(newCondition, newTrueValue, newFalseValue);
    }

    @Override
    public LIRNode visitScalarCast(ScalarCast node) {
        ScalarExpr newInput = (ScalarExpr) node.input().accept(this);
        if (newInput == node.input()) {
            return node;
        }
        return new ScalarCast(newInput, node.targetType());
    }

    @Override
    public LIRNode visitScalarLoad(ScalarLoad node) {
        IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
        if (newOffset == node.offset()) {
            return node;
        }
        return new ScalarLoad(node.buffer(), newOffset);
    }

    @Override
    public LIRNode visitScalarInput(ScalarInput node) {
        // ScalarInput has no children to rewrite
        return node;
    }

    @Override
    public LIRNode visitScalarFromIndex(ScalarFromIndex node) {
        IndexExpr newIndex = (IndexExpr) node.index().accept(this);
        if (newIndex == node.index()) {
            return node;
        }
        return new ScalarFromIndex(newIndex);
    }

    // Memory access

    @Override
    public LIRNode visitBufferRef(BufferRef node) {
        return node;
    }

    @Override
    public LIRNode visitLoad(Load node) {
        IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
        if (newOffset == node.offset()) {
            return node;
        }
        return new Load(node.buffer(), newOffset);
    }

    @Override
    public LIRNode visitStore(Store node) {
        IndexExpr newOffset = (IndexExpr) node.offset().accept(this);
        ScalarExpr newValue = (ScalarExpr) node.value().accept(this);
        if (newOffset == node.offset() && newValue == node.value()) {
            return node;
        }
        return new Store(node.buffer(), newOffset, newValue);
    }

    // Accumulators

    @Override
    public LIRNode visitAccumulator(Accumulator node) {
        return node;
    }

    @Override
    public LIRNode visitAccumulatorRead(AccumulatorRead node) {
        return node;
    }

    @Override
    public LIRNode visitAccumulatorUpdate(AccumulatorUpdate node) {
        ScalarExpr newValue = (ScalarExpr) node.value().accept(this);
        if (newValue == node.value()) {
            return node;
        }
        return new AccumulatorUpdate(node.name(), newValue);
    }

    // Loops and control flow

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
    public LIRNode visitLoopNest(LoopNest node) {
        List<Loop> newLoops = new ArrayList<>(node.loops().size());
        boolean changed = false;
        for (Loop loop : node.loops()) {
            Loop newLoop = (Loop) visitLoop(loop);
            newLoops.add(newLoop);
            if (newLoop != loop) {
                changed = true;
            }
        }
        LIRNode newBody = node.body().accept(this);
        if (!changed && newBody == node.body()) {
            return node;
        }
        return new LoopNest(newLoops, newBody);
    }

    @Override
    public LIRNode visitBlock(Block node) {
        List<LIRNode> newStatements = new ArrayList<>(node.statements().size());
        boolean changed = false;
        for (LIRNode stmt : node.statements()) {
            LIRNode newStmt = stmt.accept(this);
            newStatements.add(newStmt);
            if (newStmt != stmt) {
                changed = true;
            }
        }
        if (!changed) {
            return node;
        }
        return new Block(newStatements);
    }
}

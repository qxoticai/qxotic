package com.qxotic.jota.ir.lir;

import java.util.ArrayList;
import java.util.List;

/** Common subexpression elimination pass for unified LIR graphs. */
public final class LIRCSEPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        LIRExprGraph exprGraph = graph.exprGraph();
        LIRExprNode body = rewrite(exprGraph, graph.body());
        if (body == graph.body()) {
            return graph;
        }
        return new LIRGraph(exprGraph, graph.inputs(), graph.outputs(), body);
    }

    @Override
    public String name() {
        return "LIRCsePass";
    }

    private LIRExprNode rewrite(LIRExprGraph exprGraph, LIRExprNode node) {
        switch (node) {
            case Block block -> {
                List<LIRExprNode> statements = block.statements();
                List<LIRExprNode> rewritten = null;
                for (int i = 0; i < statements.size(); i++) {
                    LIRExprNode stmt = statements.get(i);
                    LIRExprNode next = rewrite(exprGraph, stmt);
                    if (next != stmt && rewritten == null) {
                        rewritten = new ArrayList<>(statements.size());
                        rewritten.addAll(statements.subList(0, i));
                    }
                    if (rewritten != null) {
                        rewritten.add(next);
                    }
                }
                if (rewritten == null) {
                    return block;
                }
                return exprGraph.block(rewritten);
            }
            case Store store -> {
                LIRExprNode offset = exprGraph.resolve(store.offset());
                LIRExprNode value = exprGraph.resolve(store.value());
                if (offset == store.offset() && value == store.value()) {
                    return store;
                }
                return exprGraph.store(store.buffer(), offset, value);
            }
            case Yield yield -> {
                List<LIRExprNode> values = yield.values();
                List<LIRExprNode> rewritten = null;
                for (int i = 0; i < values.size(); i++) {
                    LIRExprNode value = values.get(i);
                    LIRExprNode next = exprGraph.resolve(value);
                    if (next != value && rewritten == null) {
                        rewritten = new ArrayList<>(values.size());
                        rewritten.addAll(values.subList(0, i));
                    }
                    if (rewritten != null) {
                        rewritten.add(next);
                    }
                }
                if (rewritten == null) {
                    return yield;
                }
                return exprGraph.yield(rewritten);
            }
            case StructuredFor loop -> {
                LIRExprNode lower = exprGraph.resolve(loop.lowerBound());
                LIRExprNode upper = exprGraph.resolve(loop.upperBound());
                LIRExprNode step = exprGraph.resolve(loop.step());

                boolean changed =
                        lower != loop.lowerBound()
                                || upper != loop.upperBound()
                                || step != loop.step();

                List<LoopIterArg> iterArgs = loop.iterArgs();
                List<LoopIterArg> updatedArgs = null;
                for (int i = 0; i < iterArgs.size(); i++) {
                    LoopIterArg arg = iterArgs.get(i);
                    LIRExprNode init = exprGraph.resolve(arg.init());
                    if (init != arg.init() && updatedArgs == null) {
                        updatedArgs = new ArrayList<>(iterArgs.size());
                        updatedArgs.addAll(iterArgs.subList(0, i));
                    }
                    if (updatedArgs != null) {
                        updatedArgs.add(new LoopIterArg(arg.name(), arg.dataType(), init));
                    }
                }

                Block body = (Block) rewrite(exprGraph, loop.body());
                if (body != loop.body()) {
                    changed = true;
                }

                if (!changed && updatedArgs == null) {
                    return loop;
                }

                return exprGraph.structuredFor(
                        loop.indexName(),
                        lower,
                        upper,
                        step,
                        updatedArgs == null ? iterArgs : updatedArgs,
                        body);
            }
            default -> {
                return node;
            }
        }
    }
}

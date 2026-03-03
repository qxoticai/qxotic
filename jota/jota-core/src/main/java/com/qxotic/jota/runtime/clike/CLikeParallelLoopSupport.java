package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.Block;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.StructuredFor;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;

public final class CLikeParallelLoopSupport {
    private CLikeParallelLoopSupport() {}

    public interface Emitter {
        boolean isConstZero(LIRExprNode node);

        boolean isConstOne(LIRExprNode node);

        String emitIndexExpr(LIRExprNode node);

        String nextTempName();

        void addLine(String line);

        void indent();

        void outdent();

        void emitNode(LIRExprNode node);
    }

    public static Emitter emitter(
            Predicate<LIRExprNode> isConstZero,
            Predicate<LIRExprNode> isConstOne,
            Function<LIRExprNode, String> emitIndexExpr,
            Supplier<String> nextTempName,
            Consumer<String> addLine,
            Runnable indent,
            Runnable outdent,
            Consumer<LIRExprNode> emitNode) {
        return new Emitter() {
            @Override
            public boolean isConstZero(LIRExprNode node) {
                return isConstZero.test(node);
            }

            @Override
            public boolean isConstOne(LIRExprNode node) {
                return isConstOne.test(node);
            }

            @Override
            public String emitIndexExpr(LIRExprNode node) {
                return emitIndexExpr.apply(node);
            }

            @Override
            public String nextTempName() {
                return nextTempName.get();
            }

            @Override
            public void addLine(String line) {
                addLine.accept(line);
            }

            @Override
            public void indent() {
                indent.run();
            }

            @Override
            public void outdent() {
                outdent.run();
            }

            @Override
            public void emitNode(LIRExprNode node) {
                emitNode.accept(node);
            }
        };
    }

    public record LinearIdSpec(String indexType, String linearExpr, String oneLiteral) {}

    public static boolean emitParallelTopLevel(
            LIRExprNode body, Emitter emitter, LinearIdSpec idSpec) {
        StructuredFor loop = topLevelLoopOrNull(body);
        if (loop == null) {
            return false;
        }
        List<StructuredFor> loops = collectPerfectNest(loop, emitter);
        if (loops == null) {
            return false;
        }

        Block innermostBody = loops.getLast().body();
        if (!CLikeSourceSupport.extractYield(innermostBody).values().isEmpty()) {
            return false;
        }

        emitLinearizedLoopBody(loops, innermostBody, emitter, idSpec);
        return true;
    }

    private static StructuredFor topLevelLoopOrNull(LIRExprNode body) {
        if (body instanceof StructuredFor structuredFor) {
            return structuredFor;
        }
        if (body instanceof Block block
                && block.statements().size() == 1
                && block.statements().getFirst() instanceof StructuredFor structuredFor) {
            return structuredFor;
        }
        return null;
    }

    private static List<StructuredFor> collectPerfectNest(StructuredFor loop, Emitter emitter) {
        List<StructuredFor> loops = new ArrayList<>();
        StructuredFor current = loop;
        while (true) {
            if (!isParallelizableLoop(current, emitter)) {
                return null;
            }
            loops.add(current);
            Block body = current.body();
            if (body.statements().size() == 1
                    && body.statements().getFirst() instanceof StructuredFor nested) {
                current = nested;
                continue;
            }
            break;
        }
        return loops;
    }

    private static boolean isParallelizableLoop(StructuredFor loop, Emitter emitter) {
        if (!loop.iterArgs().isEmpty()) {
            return false;
        }
        return emitter.isConstZero(loop.lowerBound()) && emitter.isConstOne(loop.step());
    }

    private static void emitLinearizedLoopBody(
            List<StructuredFor> loops, Block innermostBody, Emitter emitter, LinearIdSpec idSpec) {
        String linear = emitter.nextTempName();
        emitter.addLine(idSpec.indexType() + " " + linear + " = " + idSpec.linearExpr() + ";");
        String total = emitter.nextTempName();
        emitter.addLine(idSpec.indexType() + " " + total + " = " + idSpec.oneLiteral() + ";");

        List<String> extents = new ArrayList<>(loops.size());
        for (StructuredFor forLoop : loops) {
            String extent = emitter.nextTempName();
            String ub = emitter.emitIndexExpr(forLoop.upperBound());
            emitter.addLine(
                    idSpec.indexType()
                            + " "
                            + extent
                            + " = ("
                            + idSpec.indexType()
                            + ")"
                            + ub
                            + ";");
            emitter.addLine(total + " *= " + extent + ";");
            extents.add(extent);
        }

        emitter.addLine("if (" + linear + " < " + total + ") {");
        emitter.indent();
        String temp = emitter.nextTempName();
        emitter.addLine(idSpec.indexType() + " " + temp + " = " + linear + ";");
        for (int i = loops.size() - 1; i >= 0; i--) {
            StructuredFor forLoop = loops.get(i);
            String extent = extents.get(i);
            String idxName = forLoop.indexName();
            emitter.addLine(
                    idSpec.indexType() + " " + idxName + " = " + temp + " % " + extent + ";");
            emitter.addLine(temp + " = " + temp + " / " + extent + ";");
        }

        int limit = innermostBody.statements().size() - 1;
        for (int i = 0; i < limit; i++) {
            emitter.emitNode(innermostBody.statements().get(i));
        }
        emitter.outdent();
        emitter.addLine("}");
    }
}

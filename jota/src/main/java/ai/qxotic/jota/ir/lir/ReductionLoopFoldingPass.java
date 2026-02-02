package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.ArrayList;
import java.util.List;

/**
 * Optimization pass that folds structured reduction loops with loop-invariant updates.
 *
 * <p>This pass targets loops of the form:
 *
 * <pre>
 * for %i = 0 to N step 1 iter_args(%acc = init) -> (dtype) {
 *   yield add %acc, %invariant
 * }
 * </pre>
 *
 * <p>Which can be folded into:
 *
 * <pre>
 * %acc = add %init, (N * %invariant)
 * </pre>
 */
public final class ReductionLoopFoldingPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        LIRNode newBody = transform(graph.body());
        if (newBody == graph.body()) {
            return graph;
        }
        return new LIRGraph(graph.inputs(), graph.outputs(), newBody);
    }

    @Override
    public String name() {
        return "ReductionLoopFolding";
    }

    private LIRNode transform(LIRNode node) {
        return switch (node) {
            case Block block -> transformBlock(block);
            case StructuredFor loop -> transformStructuredFor(loop);
            case Loop loop -> {
                LIRNode newBody = transform(loop.body());
                if (newBody == loop.body()) yield loop;
                yield new Loop(loop.indexName(), loop.bound(), loop.isParallel(), newBody);
            }
            default -> node;
        };
    }

    private LIRNode transformBlock(Block block) {
        List<LIRNode> statements = new ArrayList<>();
        boolean changed = false;
        for (LIRNode stmt : block.statements()) {
            LIRNode newStmt = transform(stmt);
            if (newStmt != stmt) {
                changed = true;
            }
            statements.add(newStmt);
        }
        if (!changed) {
            return block;
        }
        return new Block(statements);
    }

    private LIRNode transformStructuredFor(StructuredFor loop) {
        LIRNode newBody = transform(loop.body());
        if (newBody != loop.body()) {
            loop =
                    new StructuredFor(
                            loop.indexName(),
                            loop.lowerBound(),
                            loop.upperBound(),
                            loop.step(),
                            loop.iterArgs(),
                            newBody);
        }

        if (loop.iterArgs().size() != 1) {
            return loop;
        }

        LoopIterArg arg = loop.iterArgs().getFirst();
        Yield yield = extractYield(loop.body());
        if (yield.values().size() != 1) {
            return loop;
        }

        ScalarExpr update = yield.values().getFirst();
        ScalarExpr invariant = extractInvariantUpdate(update, arg.name());
        if (invariant == null || dependsOnIndex(invariant, loop.indexName())) {
            return loop;
        }

        ScalarExpr tripCount = indexToScalar(tripCountExpr(loop), arg.dataType());
        ScalarExpr scaled = new ScalarBinary(BinaryOperator.MULTIPLY, invariant, tripCount);
        ScalarExpr folded = new ScalarBinary(BinaryOperator.ADD, arg.init(), scaled);
        return new ScalarLet(arg.name(), folded);
    }

    private Yield extractYield(LIRNode body) {
        if (body instanceof Yield yield) {
            return yield;
        }
        if (body instanceof Block block && !block.statements().isEmpty()) {
            LIRNode last = block.statements().getLast();
            if (last instanceof Yield yield) {
                return yield;
            }
        }
        throw new IllegalStateException("Structured loop body must end with Yield");
    }

    private ScalarExpr extractInvariantUpdate(ScalarExpr expr, String accName) {
        if (expr instanceof ScalarBinary bin && bin.op() == BinaryOperator.ADD) {
            if (bin.left() instanceof ScalarRef ref && ref.name().equals(accName)) {
                return bin.right();
            }
            if (bin.right() instanceof ScalarRef ref && ref.name().equals(accName)) {
                return bin.left();
            }
        }
        return null;
    }

    private IndexExpr tripCountExpr(StructuredFor loop) {
        IndexExpr span = IndexBinary.subtract(loop.upperBound(), loop.lowerBound());
        return IndexBinary.divide(span, loop.step());
    }

    private ScalarExpr indexToScalar(IndexExpr index, DataType targetType) {
        if (index instanceof IndexConst constant) {
            return ScalarLiteral.of(constant.value(), targetType);
        }
        ScalarExpr asI64 = new ScalarFromIndex(index);
        return new ScalarCast(asI64, targetType);
    }

    private boolean dependsOnIndex(ScalarExpr expr, String indexName) {
        return switch (expr) {
            case ScalarLiteral ignored -> false;
            case ScalarInput ignored -> false;
            case ScalarRef ignored -> false;
            case ScalarFromIndex sfi -> dependsOnIndexExpr(sfi.index(), indexName);
            case ScalarLoad load -> dependsOnIndexExpr(load.offset(), indexName);
            case ScalarUnary unary -> dependsOnIndex(unary.input(), indexName);
            case ScalarBinary binary ->
                    dependsOnIndex(binary.left(), indexName)
                            || dependsOnIndex(binary.right(), indexName);
            case ScalarTernary ternary ->
                    dependsOnIndex(ternary.condition(), indexName)
                            || dependsOnIndex(ternary.trueValue(), indexName)
                            || dependsOnIndex(ternary.falseValue(), indexName);
            case ScalarCast cast -> dependsOnIndex(cast.input(), indexName);
        };
    }

    private boolean dependsOnIndexExpr(IndexExpr expr, String indexName) {
        return switch (expr) {
            case IndexConst ignored -> false;
            case IndexVar var -> var.name().equals(indexName);
            case IndexBinary binary ->
                    dependsOnIndexExpr(binary.left(), indexName)
                            || dependsOnIndexExpr(binary.right(), indexName);
        };
    }
}

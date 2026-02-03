package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.lir.IndexBinary.IndexBinaryOp;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * LIR v2 worklist pass that builds a unique-node expression DAG, canonicalizes it, and
 * reconstructs LIR expressions.
 */
public final class LIRV2WorklistPass implements LIRPass {

    @Override
    public LIRGraph run(LIRGraph graph) {
        LirV2Graph v2 = new LirV2Graph();
        V2Builder builder = new V2Builder(v2);
        builder.collect(graph.body());

        v2.processWorklist();

        V2ToLirRewriter rewriter = new V2ToLirRewriter(v2, builder);
        LIRNode newBody = rewriter.rewrite(graph.body());
        return new LIRGraph(graph.inputs(), graph.outputs(), newBody);
    }

    @Override
    public String name() {
        return "LIRV2WorklistPass";
    }

    private static final class V2ToLirRewriter {
        private final LirV2Graph graph;
        private final V2Builder builder;
        private final Map<V2Node, ScalarExpr> scalarCache = new IdentityHashMap<>();
        private final Map<V2Node, IndexExpr> indexCache = new IdentityHashMap<>();

        V2ToLirRewriter(LirV2Graph graph, V2Builder builder) {
            this.graph = graph;
            this.builder = builder;
        }

        LIRNode rewrite(LIRNode node) {
            return switch (node) {
                case Store store ->
                        new Store(
                                store.buffer(),
                                toIndex(builder.toIndex(store.offset())),
                                toScalar(builder.toScalar(store.value())));
                case Yield yield -> {
                    List<ScalarExpr> values =
                            yield.values().stream().map(v -> toScalar(builder.toScalar(v))).toList();
                    yield new Yield(values);
                }
                case Block block -> {
                    List<LIRNode> statements =
                            block.statements().stream().map(this::rewrite).toList();
                    yield new Block(statements);
                }
                case StructuredFor loop ->
                        new StructuredFor(
                                loop.indexName(),
                                toIndex(builder.toIndex(loop.lowerBound())),
                                toIndex(builder.toIndex(loop.upperBound())),
                                toIndex(builder.toIndex(loop.step())),
                                loop.iterArgs().stream()
                                        .map(arg ->
                                                new LoopIterArg(
                                                        arg.name(),
                                                        arg.dataType(),
                                                        toScalar(builder.toScalar(arg.init()))))
                                        .toList(),
                                rewrite(loop.body()));
                case TiledLoop tiled ->
                        new TiledLoop(
                                tiled.outerName(),
                                tiled.innerName(),
                                toIndex(builder.toIndex(tiled.totalBound())),
                                tiled.tileSize(),
                                rewrite(tiled.body()));
                case ScalarLet let ->
                        new ScalarLet(
                                let.name(), toScalar(builder.toScalar(let.value())));
                default -> node;
            };
        }

        private ScalarExpr toScalar(V2Node node) {
            V2Node resolved = graph.resolve(node);
            ScalarExpr cached = scalarCache.get(resolved);
            if (cached != null) {
                return cached;
            }

            ScalarExpr expr =
                    switch (resolved.kind()) {
                        case S_CONST ->
                                ScalarLiteral.ofRawBits(
                                        ((SConst) resolved).rawBits(), resolved.dataType());
                        case S_INPUT -> new ScalarInput(((SInput) resolved).inputId(), resolved.dataType());
                        case S_UNARY ->
                                new ScalarUnary(
                                        ((SUnary) resolved).op(),
                                        toScalar(((SUnary) resolved).input()));
                        case S_BINARY ->
                                new ScalarBinary(
                                        ((SBinary) resolved).op(),
                                        toScalar(((SBinary) resolved).left()),
                                        toScalar(((SBinary) resolved).right()));
                        case S_TERNARY ->
                                new ScalarTernary(
                                        toScalar(((STernary) resolved).condition()),
                                        toScalar(((STernary) resolved).trueValue()),
                                        toScalar(((STernary) resolved).falseValue()));
                        case S_CAST ->
                                new ScalarCast(
                                        toScalar(((SCast) resolved).input()),
                                        ((SCast) resolved).targetType());
                        case S_LOAD ->
                                new ScalarLoad(
                                        ((SLoad) resolved).buffer(),
                                        toIndex(((SLoad) resolved).offset()));
                        case S_FROM_INDEX ->
                                new ScalarFromIndex(toIndex(((SFromIndex) resolved).indexExpr()));
                        case S_REF -> new ScalarRef(((SRef) resolved).name(), resolved.dataType());
                        case I_CONST,
                                I_VAR,
                                I_BINARY ->
                                throw new IllegalStateException("Expected scalar node, got " + resolved.kind());
                    };

            scalarCache.put(resolved, expr);
            return expr;
        }

        private IndexExpr toIndex(V2Node node) {
            V2Node resolved = graph.resolve(node);
            IndexExpr cached = indexCache.get(resolved);
            if (cached != null) {
                return cached;
            }

            IndexExpr expr =
                    switch (resolved.kind()) {
                        case I_CONST -> IndexConst.of(((IConst) resolved).value());
                        case I_VAR -> new IndexVar(((IVar) resolved).name());
                        case I_BINARY ->
                                new IndexBinary(
                                        ((IBinary) resolved).op(),
                                        toIndex(((IBinary) resolved).left()),
                                        toIndex(((IBinary) resolved).right()));
                        case S_CONST,
                                S_INPUT,
                                S_UNARY,
                                S_BINARY,
                                S_TERNARY,
                                S_CAST,
                                S_LOAD,
                                S_FROM_INDEX,
                                S_REF ->
                                throw new IllegalStateException("Expected index node, got " + resolved.kind());
                    };

            indexCache.put(resolved, expr);
            return expr;
        }
    }
}

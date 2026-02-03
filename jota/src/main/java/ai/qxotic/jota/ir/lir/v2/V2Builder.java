package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.*;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public final class V2Builder {
    private final LirV2Graph graph;
    private final Map<ScalarExpr, V2Node> scalarMap = new IdentityHashMap<>();
    private final Map<IndexExpr, V2Node> indexMap = new IdentityHashMap<>();
    private final Map<ExprKey, V2Node> scalarKeyMap = new HashMap<>();
    private final Map<ExprKey, V2Node> indexKeyMap = new HashMap<>();
    private final Map<ScalarExpr, ExprKey> scalarKeyCache = new IdentityHashMap<>();
    private final Map<IndexExpr, ExprKey> indexKeyCache = new IdentityHashMap<>();
    private final Deque<Map<String, V2Node>> scalarLetScopes = new ArrayDeque<>();

    public V2Builder(LirV2Graph graph) {
        this.graph = graph;
    }

    public void collect(LIRNode node) {
        switch (node) {
            case Store store -> {
                toIndex(store.offset());
                toScalar(store.value());
            }
            case Yield yield -> {
                for (ScalarExpr value : yield.values()) {
                    toScalar(value);
                }
            }
            case Block block -> {
                pushScope();
                for (LIRNode stmt : block.statements()) {
                    collect(stmt);
                }
                popScope();
            }
            case StructuredFor loop -> {
                toIndex(loop.lowerBound());
                toIndex(loop.upperBound());
                toIndex(loop.step());
                for (LoopIterArg arg : loop.iterArgs()) {
                    toScalar(arg.init());
                }
                pushScope();
                for (LoopIterArg arg : loop.iterArgs()) {
                    defineLet(arg.name(), graph.scalarRef(arg.name(), arg.dataType()));
                }
                collect(loop.body());
                popScope();
            }
            case TiledLoop tiled -> {
                toIndex(tiled.totalBound());
                pushScope();
                collect(tiled.body());
                popScope();
            }
            case ScalarLet let -> {
                V2Node value = toScalar(let.value());
                defineLet(let.name(), value);
            }
            default -> {}
        }
    }

    public V2Node toScalar(ScalarExpr expr) {
        V2Node cached = scalarMap.get(expr);
        if (cached != null) {
            return cached;
        }
        if (expr instanceof ScalarLoad load) {
            V2Node node = graph.scalarLoad(load.buffer(), toIndex(load.offset()), load.dataType());
            scalarMap.put(expr, node);
            return node;
        }
        ExprKey key = scalarKeyFor(expr);
        V2Node existing = scalarKeyMap.get(key);
        if (existing != null) {
            scalarMap.put(expr, existing);
            return existing;
        }
        V2Node node = switch (expr) {
            case ScalarLiteral literal -> graph.scalarConst(literal.rawBits(), literal.dataType());
            case ScalarInput input -> graph.scalarInput(input.id(), input.dataType());
            case ScalarUnary unary -> graph.scalarUnary(unary.op(), toScalar(unary.input()));
            case ScalarBinary binary ->
                    graph.scalarBinary(
                            binary.op(), toScalar(binary.left()), toScalar(binary.right()));
            case ScalarTernary ternary ->
                    graph.scalarTernary(
                            toScalar(ternary.condition()),
                            toScalar(ternary.trueValue()),
                            toScalar(ternary.falseValue()));
            case ScalarCast cast -> graph.scalarCast(toScalar(cast.input()), cast.targetType());
            case ScalarLoad load -> graph.scalarLoad(load.buffer(), toIndex(load.offset()), load.dataType());
            case ScalarFromIndex fromIndex -> graph.scalarFromIndex(toIndex(fromIndex.index()), DataType.I64);
            case ScalarRef ref -> lookupLet(ref.name(), ref.dataType());
        };
        scalarMap.put(expr, node);
        if (node instanceof SLoad) {
            return node;
        }
        scalarKeyMap.put(key, node);
        return node;
    }

    public V2Node toIndex(IndexExpr expr) {
        V2Node cached = indexMap.get(expr);
        if (cached != null) {
            return cached;
        }
        ExprKey key = indexKeyFor(expr);
        V2Node existing = indexKeyMap.get(key);
        if (existing != null) {
            indexMap.put(expr, existing);
            return existing;
        }
        V2Node node = switch (expr) {
            case IndexConst constant -> graph.indexConst(constant.value());
            case IndexVar var -> graph.indexVar(var.name());
            case IndexBinary binary ->
                    graph.indexBinary(binary.op(), toIndex(binary.left()), toIndex(binary.right()));
        };
        indexMap.put(expr, node);
        indexKeyMap.put(key, node);
        return node;
    }

    private void pushScope() {
        scalarLetScopes.push(new HashMap<>());
    }

    private void popScope() {
        scalarLetScopes.pop();
    }

    private void defineLet(String name, V2Node value) {
        if (scalarLetScopes.isEmpty()) {
            scalarLetScopes.push(new HashMap<>());
        }
        scalarLetScopes.peek().put(name, value);
    }

    private V2Node lookupLet(String name, DataType dataType) {
        for (Map<String, V2Node> scope : scalarLetScopes) {
            V2Node value = scope.get(name);
            if (value != null) {
                return value;
            }
        }
        return graph.scalarRef(name, dataType);
    }

    private ExprKey scalarKeyFor(ScalarExpr expr) {
        ExprKey cached = scalarKeyCache.get(expr);
        if (cached != null) {
            return cached;
        }
        ExprKey key =
                switch (expr) {
                    case ScalarLiteral literal ->
                            new ExprKey(
                                    "lit",
                                    literal.dataType(),
                                    literal.rawBits(),
                                    java.util.List.of());
                    case ScalarInput input ->
                            new ExprKey(
                                    "input",
                                    input.dataType(),
                                    input.id(),
                                    java.util.List.of());
                    case ScalarUnary unary ->
                            new ExprKey(
                                    "unary",
                                    unary.dataType(),
                                    unary.op(),
                                    java.util.List.of(scalarKeyFor(unary.input())));
                    case ScalarBinary binary ->
                            new ExprKey(
                                    "binary",
                                    binary.dataType(),
                                    binary.op(),
                                    java.util.List.of(
                                            scalarKeyFor(binary.left()),
                                            scalarKeyFor(binary.right())));
                    case ScalarTernary ternary ->
                            new ExprKey(
                                    "ternary",
                                    ternary.dataType(),
                                    null,
                                    java.util.List.of(
                                            scalarKeyFor(ternary.condition()),
                                            scalarKeyFor(ternary.trueValue()),
                                            scalarKeyFor(ternary.falseValue())));
                    case ScalarCast cast ->
                            new ExprKey(
                                    "cast",
                                    cast.targetType(),
                                    cast.targetType(),
                                    java.util.List.of(scalarKeyFor(cast.input())));
                    case ScalarFromIndex fromIndex ->
                            new ExprKey(
                                    "fromIndex",
                                    DataType.I64,
                                    null,
                                    java.util.List.of(indexKeyFor(fromIndex.index())));
                    case ScalarRef ref ->
                            new ExprKey(
                                    "ref",
                                    ref.dataType(),
                                    ref.name(),
                                    java.util.List.of());
                    case ScalarLoad load ->
                            new ExprKey(
                                    "load",
                                    load.dataType(),
                                    load.buffer().id(),
                                    java.util.List.of(indexKeyFor(load.offset())));
                };
        scalarKeyCache.put(expr, key);
        return key;
    }

    private ExprKey indexKeyFor(IndexExpr expr) {
        ExprKey cached = indexKeyCache.get(expr);
        if (cached != null) {
            return cached;
        }
        ExprKey key =
                switch (expr) {
                    case IndexConst constant ->
                            new ExprKey("iconst", DataType.I64, constant.value(), java.util.List.of());
                    case IndexVar var ->
                            new ExprKey("ivar", DataType.I64, var.name(), java.util.List.of());
                    case IndexBinary binary ->
                            new ExprKey(
                                    "ibinary",
                                    DataType.I64,
                                    binary.op(),
                                    java.util.List.of(
                                            indexKeyFor(binary.left()),
                                            indexKeyFor(binary.right())));
                };
        indexKeyCache.put(expr, key);
        return key;
    }

    private static final class ExprKey {
        private final String kind;
        private final DataType dataType;
        private final Object attr;
        private final java.util.List<ExprKey> children;
        private final int hash;

        private ExprKey(
                String kind, DataType dataType, Object attr, java.util.List<ExprKey> children) {
            this.kind = kind;
            this.dataType = dataType;
            this.attr = attr;
            this.children = children;
            this.hash = computeHash();
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof ExprKey that)) {
                return false;
            }
            return kind.equals(that.kind)
                    && dataType == that.dataType
                    && java.util.Objects.equals(attr, that.attr)
                    && children.equals(that.children);
        }

        private int computeHash() {
            int result = kind.hashCode();
            result = 31 * result + dataType.hashCode();
            result = 31 * result + (attr == null ? 0 : attr.hashCode());
            result = 31 * result + children.hashCode();
            return result;
        }
    }
}

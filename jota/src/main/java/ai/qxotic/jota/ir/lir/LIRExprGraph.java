package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.BufferRef;
import ai.qxotic.jota.ir.lir.IndexBinaryOp;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public final class LIRExprGraph {
    private final Map<NodeKey, LIRExprNode> uniqueNodes = new HashMap<>();
    private final ArrayDeque<LIRExprNode> worklist = new ArrayDeque<>();
    private int nextId;

    public LIRExprNode resolve(LIRExprNode node) {
        LIRExprNode current = node;
        while (current.replacement() != null) {
            current = current.replacement();
        }
        return current;
    }

    public void processWorklist() {
        while (!worklist.isEmpty()) {
            LIRExprNode node = resolve(worklist.removeFirst());
            LIRExprNode canonical = node.canonicalize(this);
            if (canonical != node) {
                replace(node, canonical);
            }
        }
    }

    private void replace(LIRExprNode oldNode, LIRExprNode newNode) {
        LIRExprNode canonicalNew = resolve(newNode);
        if (oldNode == canonicalNew) {
            return;
        }

        Use use = oldNode.uses();
        while (use != null) {
            Use next = use.next;
            LIRExprNode user = use.user;
            int inputIndex = use.inputIndex;
            user.replaceInput(inputIndex, canonicalNew);
            canonicalNew.addUse(user, inputIndex);
            enqueue(user);
            use = next;
        }
        oldNode.clearUses();
        oldNode.setReplacement(canonicalNew);
    }

    private void enqueue(LIRExprNode node) {
        worklist.addLast(node);
    }

    private LIRExprNode registerNode(LIRExprNode node) {
        LIRExprNode[] inputs = node.inputs();
        for (int i = 0; i < inputs.length; i++) {
            inputs[i].addUse(node, i);
        }
        enqueue(node);
        return node;
    }

    private LIRExprNode addUnique(LIRExprNode node) {
        NodeKey key = NodeKey.of(node);
        LIRExprNode existing = uniqueNodes.get(key);
        if (existing != null) {
            return existing;
        }
        uniqueNodes.put(key, node);
        return registerNode(node);
    }

    private LIRExprNode addNode(LIRExprNode node) {
        return registerNode(node);
    }

    public SConst scalarConst(long rawBits, DataType dataType) {
        return (SConst) addUnique(new SConst(nextId++, rawBits, dataType));
    }

    public SInput scalarInput(int inputId, DataType dataType) {
        return (SInput) addUnique(new SInput(nextId++, inputId, dataType));
    }

    public SUnary scalarUnary(UnaryOperator op, LIRExprNode input) {
        DataType resultType = resultTypeUnary(op, input.dataType());
        return (SUnary) addUnique(new SUnary(nextId++, op, input, resultType));
    }

    public SBinary scalarBinary(BinaryOperator op, LIRExprNode left, LIRExprNode right) {
        if (isCommutative(op) && left.id() > right.id()) {
            LIRExprNode tmp = left;
            left = right;
            right = tmp;
        }
        DataType resultType = resultTypeBinary(op, left.dataType());
        return (SBinary) addUnique(new SBinary(nextId++, op, left, right, resultType));
    }

    public STernary scalarTernary(LIRExprNode cond, LIRExprNode trueValue, LIRExprNode falseValue) {
        return (STernary)
                addUnique(
                        new STernary(nextId++, cond, trueValue, falseValue, trueValue.dataType()));
    }

    public SCast scalarCast(LIRExprNode input, DataType targetType) {
        return (SCast) addUnique(new SCast(nextId++, input, targetType));
    }

    public SLoad scalarLoad(BufferRef buffer, LIRExprNode offset, DataType dataType) {
        return (SLoad) addNode(new SLoad(nextId++, buffer, offset, dataType));
    }

    public SFromIndex scalarFromIndex(LIRExprNode indexExpr, DataType dataType) {
        return (SFromIndex) addUnique(new SFromIndex(nextId++, indexExpr, dataType));
    }

    public SRef scalarRef(String name, DataType dataType) {
        return (SRef) addUnique(new SRef(nextId++, name, dataType));
    }

    public IConst indexConst(long value) {
        return (IConst) addUnique(new IConst(nextId++, value));
    }

    public IVar indexVar(String name) {
        return (IVar) addUnique(new IVar(nextId++, name));
    }

    public IBinary indexBinary(IndexBinaryOp op, LIRExprNode left, LIRExprNode right) {
        if (isCommutative(op) && left.id() > right.id()) {
            LIRExprNode tmp = left;
            left = right;
            right = tmp;
        }
        return (IBinary) addUnique(new IBinary(nextId++, op, left, right));
    }

    public Block block(java.util.List<LIRExprNode> statements) {
        return (Block) addNode(new Block(nextId++, statements));
    }

    public Store store(BufferRef buffer, LIRExprNode offset, LIRExprNode value) {
        return (Store) addNode(new Store(nextId++, buffer, offset, value));
    }

    public Yield yield(java.util.List<LIRExprNode> values) {
        return (Yield) addNode(new Yield(nextId++, values));
    }

    public StructuredFor structuredFor(
            String indexName,
            LIRExprNode lowerBound,
            LIRExprNode upperBound,
            LIRExprNode step,
            java.util.List<LoopIterArg> iterArgs,
            Block body) {
        return (StructuredFor)
                addNode(
                        new StructuredFor(
                                nextId++, indexName, lowerBound, upperBound, step, iterArgs, body));
    }

    public TiledLoop tiledLoop(
            String outerName, String innerName, LIRExprNode totalBound, long tileSize, Block body) {
        return (TiledLoop)
                addNode(new TiledLoop(nextId++, outerName, innerName, totalBound, tileSize, body));
    }

    LIRExprNode foldUnary(UnaryOperator op, SConst input) {
        DataType type = input.dataType();
        long bits = input.rawBits();

        switch (op) {
            case LOGICAL_NOT -> {
                return scalarConst(booleanBits(!toBoolean(bits, type)), DataType.BOOL);
            }
            case BITWISE_NOT -> {
                return scalarConst(~bitsForInt(bits, type), type);
            }
            case NEGATE -> {
                return scalarConst(fromDouble(-toDouble(bits, type), type), type);
            }
            case ABS -> {
                return scalarConst(fromDouble(Math.abs(toDouble(bits, type)), type), type);
            }
            case EXP -> {
                return scalarConst(fromDouble(Math.exp(toDouble(bits, type)), type), type);
            }
            case LOG -> {
                return scalarConst(fromDouble(Math.log(toDouble(bits, type)), type), type);
            }
            case SQRT -> {
                return scalarConst(fromDouble(Math.sqrt(toDouble(bits, type)), type), type);
            }
            case SQUARE -> {
                double value = toDouble(bits, type);
                return scalarConst(fromDouble(value * value, type), type);
            }
            case SIN -> {
                return scalarConst(fromDouble(Math.sin(toDouble(bits, type)), type), type);
            }
            case COS -> {
                return scalarConst(fromDouble(Math.cos(toDouble(bits, type)), type), type);
            }
            case TAN -> {
                return scalarConst(fromDouble(Math.tan(toDouble(bits, type)), type), type);
            }
            case TANH -> {
                return scalarConst(fromDouble(Math.tanh(toDouble(bits, type)), type), type);
            }
            case RECIPROCAL -> {
                return scalarConst(fromDouble(1.0 / toDouble(bits, type), type), type);
            }
        }
        return input;
    }

    LIRExprNode foldBinary(BinaryOperator op, SConst left, SConst right) {
        DataType type = left.dataType();
        long l = left.rawBits();
        long r = right.rawBits();
        return switch (op) {
            case LOGICAL_AND ->
                    scalarConst(
                            booleanBits(toBoolean(l, type) && toBoolean(r, type)), DataType.BOOL);
            case LOGICAL_OR ->
                    scalarConst(
                            booleanBits(toBoolean(l, type) || toBoolean(r, type)), DataType.BOOL);
            case LOGICAL_XOR ->
                    scalarConst(
                            booleanBits(toBoolean(l, type) ^ toBoolean(r, type)), DataType.BOOL);
            case BITWISE_AND -> scalarConst(bitsForInt(l, type) & bitsForInt(r, type), type);
            case BITWISE_OR -> scalarConst(bitsForInt(l, type) | bitsForInt(r, type), type);
            case BITWISE_XOR -> scalarConst(bitsForInt(l, type) ^ bitsForInt(r, type), type);
            case EQUAL -> scalarConst(booleanBits(compare(l, r, type) == 0), DataType.BOOL);
            case LESS_THAN -> scalarConst(booleanBits(compare(l, r, type) < 0), DataType.BOOL);
            case MIN ->
                    scalarConst(
                            fromDouble(Math.min(toDouble(l, type), toDouble(r, type)), type), type);
            case MAX ->
                    scalarConst(
                            fromDouble(Math.max(toDouble(l, type), toDouble(r, type)), type), type);
            case ADD -> scalarConst(fromDouble(toDouble(l, type) + toDouble(r, type), type), type);
            case SUBTRACT ->
                    scalarConst(fromDouble(toDouble(l, type) - toDouble(r, type), type), type);
            case MULTIPLY ->
                    scalarConst(fromDouble(toDouble(l, type) * toDouble(r, type), type), type);
            case DIVIDE -> scalarConst(fromDouble(toDouble(l, type) / toDouble(r, type), type), type);
            case POW ->
                    scalarConst(
                            fromDouble(Math.pow(toDouble(l, type), toDouble(r, type)), type), type);
        };
    }

    LIRExprNode foldTernary(SConst condition, LIRExprNode trueValue, LIRExprNode falseValue) {
        boolean cond = toBoolean(condition.rawBits(), condition.dataType());
        return cond ? trueValue : falseValue;
    }

    LIRExprNode foldCast(SConst input, DataType targetType) {
        if (input.dataType() == targetType) {
            return input;
        }
        return scalarConst(fromDouble(toDouble(input.rawBits(), input.dataType()), targetType), targetType);
    }

    LIRExprNode foldIndexBinary(IndexBinaryOp op, IConst left, IConst right) {
        long l = left.value();
        long r = right.value();
        return switch (op) {
            case ADD -> indexConst(l + r);
            case SUBTRACT -> indexConst(l - r);
            case MULTIPLY -> indexConst(l * r);
            case DIVIDE -> indexConst(l / r);
            case MODULO -> indexConst(l % r);
            case BITWISE_AND -> indexConst(l & r);
            case SHIFT_LEFT -> indexConst(l << r);
            case SHIFT_RIGHT -> indexConst(l >> r);
        };
    }

    LIRExprNode simplifyIndexBinary(IndexBinaryOp op, LIRExprNode left, LIRExprNode right) {
        if (left instanceof IConst leftConst) {
            long l = leftConst.value();
            if (op == IndexBinaryOp.ADD && l == 0) {
                return right;
            }
            if (op == IndexBinaryOp.MULTIPLY && l == 1) {
                return right;
            }
            if (op == IndexBinaryOp.MULTIPLY && l == 0) {
                return leftConst;
            }
        }

        if (right instanceof IConst rightConst) {
            long r = rightConst.value();
            if (op == IndexBinaryOp.ADD && r == 0) {
                return left;
            }
            if (op == IndexBinaryOp.SUBTRACT && r == 0) {
                return left;
            }
            if (op == IndexBinaryOp.MULTIPLY && r == 1) {
                return left;
            }
            if (op == IndexBinaryOp.MULTIPLY && r == 0) {
                return rightConst;
            }
            if (op == IndexBinaryOp.DIVIDE && r == 1) {
                return left;
            }
            if (op == IndexBinaryOp.MODULO && r == 1) {
                return indexConst(0);
            }
            if (op == IndexBinaryOp.SHIFT_LEFT && r == 0) {
                return left;
            }
            if (op == IndexBinaryOp.SHIFT_RIGHT && r == 0) {
                return left;
            }
        }

        return indexBinary(op, left, right);
    }

    LIRExprNode simplifyBinary(BinaryOperator op, LIRExprNode left, LIRExprNode right) {
        if (left instanceof SConst leftConst) {
            if (isZero(leftConst) && op == BinaryOperator.ADD) {
                return right;
            }
            if (isZero(leftConst) && op == BinaryOperator.MULTIPLY) {
                return leftConst;
            }
            if (isOne(leftConst) && op == BinaryOperator.MULTIPLY) {
                return right;
            }
            if (op == BinaryOperator.LOGICAL_OR && leftConst.dataType() == DataType.BOOL) {
                return toBoolean(leftConst.rawBits(), DataType.BOOL)
                        ? leftConst
                        : right;
            }
            if (op == BinaryOperator.LOGICAL_AND && leftConst.dataType() == DataType.BOOL) {
                return toBoolean(leftConst.rawBits(), DataType.BOOL)
                        ? right
                        : leftConst;
            }
        }

        if (right instanceof SConst rightConst) {
            if (isZero(rightConst) && op == BinaryOperator.ADD) {
                return left;
            }
            if (isZero(rightConst) && op == BinaryOperator.SUBTRACT) {
                return left;
            }
            if (isZero(rightConst) && op == BinaryOperator.MULTIPLY) {
                return rightConst;
            }
            if (isOne(rightConst) && op == BinaryOperator.MULTIPLY) {
                return left;
            }
            if (isOne(rightConst) && op == BinaryOperator.DIVIDE) {
                return left;
            }
            if (op == BinaryOperator.LOGICAL_OR && rightConst.dataType() == DataType.BOOL) {
                return toBoolean(rightConst.rawBits(), DataType.BOOL)
                        ? rightConst
                        : left;
            }
            if (op == BinaryOperator.LOGICAL_AND && rightConst.dataType() == DataType.BOOL) {
                return toBoolean(rightConst.rawBits(), DataType.BOOL)
                        ? left
                        : rightConst;
            }
        }

        return scalarBinary(op, left, right);
    }

    private static boolean isCommutative(BinaryOperator op) {
        return switch (op) {
            case ADD,
                    MULTIPLY,
                    MIN,
                    MAX,
                    LOGICAL_AND,
                    LOGICAL_OR,
                    LOGICAL_XOR,
                    BITWISE_AND,
                    BITWISE_OR,
                    BITWISE_XOR,
                    EQUAL -> true;
            default -> false;
        };
    }

    private static boolean isCommutative(IndexBinaryOp op) {
        return switch (op) {
            case ADD, MULTIPLY -> true;
            default -> false;
        };
    }

    private static DataType resultTypeUnary(UnaryOperator op, DataType input) {
        return switch (op) {
            case LOGICAL_NOT -> DataType.BOOL;
            default -> input;
        };
    }

    private static DataType resultTypeBinary(BinaryOperator op, DataType left) {
        return switch (op) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR, EQUAL, LESS_THAN -> DataType.BOOL;
            default -> left;
        };
    }

    private static boolean toBoolean(long bits, DataType type) {
        if (type == DataType.BOOL) {
            return bits != 0;
        }
        if (type == DataType.I8) {
            return ((byte) bits) != 0;
        }
        if (type == DataType.I16) {
            return ((short) bits) != 0;
        }
        if (type == DataType.I32) {
            return ((int) bits) != 0;
        }
        if (type == DataType.I64) {
            return bits != 0;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits) != 0.0f;
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits) != 0.0f;
        }
        if (type == DataType.FP64) {
            return Double.longBitsToDouble(bits) != 0.0;
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits) != 0.0f;
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static long bitsForInt(long bits, DataType type) {
        if (type == DataType.I8) {
            return (byte) bits;
        }
        if (type == DataType.I16) {
            return (short) bits;
        }
        if (type == DataType.I32) {
            return (int) bits;
        }
        if (type == DataType.I8) {
            return (byte) bits;
        }
        if (type == DataType.I16) {
            return (short) bits;
        }
        if (type == DataType.I64) {
            return bits;
        }
        if (type == DataType.BOOL) {
            return bits != 0 ? 1L : 0L;
        }
        return bits;
    }

    private static long booleanBits(boolean value) {
        return value ? 1L : 0L;
    }

    private static boolean isZero(SConst constant) {
        DataType type = constant.dataType();
        long bits = constant.rawBits();
        if (type == DataType.BOOL) {
            return bits == 0;
        }
        if (type == DataType.I8) {
            return (byte) bits == 0;
        }
        if (type == DataType.I16) {
            return (short) bits == 0;
        }
        if (type == DataType.I32) {
            return (int) bits == 0;
        }
        if (type == DataType.I64) {
            return bits == 0L;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits) == 0.0f;
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits) == 0.0f;
        }
        if (type == DataType.FP64) {
            return Double.longBitsToDouble(bits) == 0.0;
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits) == 0.0f;
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static boolean isOne(SConst constant) {
        DataType type = constant.dataType();
        long bits = constant.rawBits();
        if (type == DataType.BOOL) {
            return bits != 0;
        }
        if (type == DataType.I8) {
            return (byte) bits == 1;
        }
        if (type == DataType.I16) {
            return (short) bits == 1;
        }
        if (type == DataType.I32) {
            return (int) bits == 1;
        }
        if (type == DataType.I64) {
            return bits == 1L;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits) == 1.0f;
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits) == 1.0f;
        }
        if (type == DataType.FP64) {
            return Double.longBitsToDouble(bits) == 1.0;
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits) == 1.0f;
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static int compare(long left, long right, DataType type) {
        if (type == DataType.BOOL) {
            return Boolean.compare(left != 0, right != 0);
        }
        if (type == DataType.I8) {
            return Byte.compare((byte) left, (byte) right);
        }
        if (type == DataType.I16) {
            return Short.compare((short) left, (short) right);
        }
        if (type == DataType.I32) {
            return Integer.compare((int) left, (int) right);
        }
        if (type == DataType.I64) {
            return Long.compare(left, right);
        }
        if (type == DataType.FP16) {
            return Float.compare(
                    Float.float16ToFloat((short) left), Float.float16ToFloat((short) right));
        }
        if (type == DataType.FP32) {
            return Float.compare(
                    Float.intBitsToFloat((int) left), Float.intBitsToFloat((int) right));
        }
        if (type == DataType.FP64) {
            return Double.compare(
                    Double.longBitsToDouble(left), Double.longBitsToDouble(right));
        }
        if (type == DataType.BF16) {
            return Float.compare(
                    BFloat16.toFloat((short) left), BFloat16.toFloat((short) right));
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static double toDouble(long bits, DataType type) {
        if (type == DataType.BOOL) {
            return bits != 0 ? 1.0 : 0.0;
        }
        if (type == DataType.I8) {
            return (byte) bits;
        }
        if (type == DataType.I16) {
            return (short) bits;
        }
        if (type == DataType.I32) {
            return (int) bits;
        }
        if (type == DataType.I64) {
            return bits;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits);
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits);
        }
        if (type == DataType.FP64) {
            return Double.longBitsToDouble(bits);
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits);
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static long fromDouble(double value, DataType type) {
        if (type == DataType.BOOL) {
            return value != 0.0 ? 1L : 0L;
        }
        if (type == DataType.I8) {
            return (byte) value;
        }
        if (type == DataType.I16) {
            return (short) value;
        }
        if (type == DataType.I32) {
            return (int) value;
        }
        if (type == DataType.I64) {
            return (long) value;
        }
        if (type == DataType.FP16) {
            return Float.floatToFloat16((float) value);
        }
        if (type == DataType.FP32) {
            return Float.floatToRawIntBits((float) value);
        }
        if (type == DataType.FP64) {
            return Double.doubleToRawLongBits(value);
        }
        if (type == DataType.BF16) {
            return BFloat16.fromFloat((float) value);
        }
        throw new IllegalArgumentException("Unsupported data type: " + type);
    }

    private static final class NodeKey {
        private final LIRExprKind kind;
        private final DataType dataType;
        private final Object attr;
        private final int[] inputs;
        private final int hash;

        private NodeKey(LIRExprKind kind, DataType dataType, Object attr, int[] inputs) {
            this.kind = kind;
            this.dataType = dataType;
            this.attr = attr;
            this.inputs = inputs;
            this.hash = computeHash(kind, dataType, attr, inputs);
        }

        static NodeKey of(LIRExprNode node) {
            Object attr = switch (node.kind()) {
                case S_CONST -> ((SConst) node).rawBits();
                case S_INPUT -> ((SInput) node).inputId();
                case S_UNARY -> ((SUnary) node).op();
                case S_BINARY -> ((SBinary) node).op();
                case S_TERNARY -> null;
                case S_CAST -> ((SCast) node).targetType();
                case S_LOAD -> ((SLoad) node).buffer();
                case S_FROM_INDEX -> null;
                case S_REF -> ((SRef) node).name();
                case I_CONST -> ((IConst) node).value();
                case I_VAR -> ((IVar) node).name();
                case I_BINARY -> ((IBinary) node).op();
                case BLOCK, STORE, YIELD, STRUCTURED_FOR, TILED_LOOP -> null;
            };

            int[] inputIds = Arrays.stream(node.inputs()).mapToInt(LIRExprNode::id).toArray();
            return new NodeKey(node.kind(), node.dataType(), attr, inputIds);
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
            if (!(other instanceof NodeKey that)) {
                return false;
            }
            return kind == that.kind
                    && dataType == that.dataType
                    && Objects.equals(attr, that.attr)
                    && Arrays.equals(inputs, that.inputs);
        }

        private static int computeHash(
                LIRExprKind kind, DataType dataType, Object attr, int[] inputs) {
            int result = kind.hashCode();
            result = 31 * result + (dataType == null ? 0 : dataType.hashCode());
            result = 31 * result + (attr == null ? 0 : attr.hashCode());
            result = 31 * result + Arrays.hashCode(inputs);
            return result;
        }
    }
}

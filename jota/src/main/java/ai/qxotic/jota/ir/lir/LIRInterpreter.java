package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.ScalarArg;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Reference interpreter for unified LIR graphs. */
public final class LIRInterpreter {

    public void execute(
            LIRGraph graph,
            List<MemoryView<?>> buffers,
            List<ScalarArg> scalars,
            List<MemoryView<?>> outputs,
            MemoryDomain<?> memoryDomain) {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(buffers, "buffers");
        Objects.requireNonNull(scalars, "scalars");
        Objects.requireNonNull(outputs, "outputs");
        Objects.requireNonNull(memoryDomain, "memoryDomain");

        Bindings bindings = Bindings.from(graph, buffers, scalars, outputs, memoryDomain);
        executeNode(graph.body(), new Env(), bindings);
    }

    private List<ScalarValue> executeNode(LIRExprNode node, Env env, Bindings bindings) {
        switch (node) {
            case Block block -> {
                for (LIRExprNode stmt : block.statements()) {
                    List<ScalarValue> yielded = executeNode(stmt, env, bindings);
                    if (yielded != null) {
                        return yielded;
                    }
                }
                return null;
            }
            case Store store -> {
                ScalarValue value = evalScalar(store.value(), env, bindings);
                long offset = evalIndex(store.offset(), env, bindings);
                MemoryView<?> view = bindings.buffer(store.buffer());
                writeScalar(view, store.buffer().dataType(), value, offset, bindings.access);
                return null;
            }
            case StructuredFor loop -> {
                long lower = evalIndex(loop.lowerBound(), env, bindings);
                long upper = evalIndex(loop.upperBound(), env, bindings);
                long step = evalIndex(loop.step(), env, bindings);
                List<LoopIterArg> iterArgs = loop.iterArgs();

                ScalarValue[] iterValues = new ScalarValue[iterArgs.size()];
                for (int i = 0; i < iterArgs.size(); i++) {
                    iterValues[i] = evalScalar(iterArgs.get(i).init(), env, bindings);
                }

                if (step == 0) {
                    throw new IllegalStateException("StructuredFor step cannot be zero");
                }

                if (step > 0) {
                    for (long idx = lower; idx < upper; idx += step) {
                        Env iterEnv = env.copy();
                        iterEnv.indexVars.put(loop.indexName(), idx);
                        for (int i = 0; i < iterArgs.size(); i++) {
                            iterEnv.scalarVars.put(iterArgs.get(i).name(), iterValues[i]);
                        }
                        List<ScalarValue> yielded = executeNode(loop.body(), iterEnv, bindings);
                        if (yielded == null) {
                            throw new IllegalStateException("Structured loop body must yield");
                        }
                        if (yielded.size() != iterArgs.size()) {
                            throw new IllegalStateException(
                                    "Yield arity "
                                            + yielded.size()
                                            + " does not match iter args "
                                            + iterArgs.size());
                        }
                        for (int i = 0; i < iterArgs.size(); i++) {
                            iterValues[i] = castValue(yielded.get(i), iterArgs.get(i).dataType());
                        }
                    }
                } else {
                    for (long idx = lower; idx > upper; idx += step) {
                        Env iterEnv = env.copy();
                        iterEnv.indexVars.put(loop.indexName(), idx);
                        for (int i = 0; i < iterArgs.size(); i++) {
                            iterEnv.scalarVars.put(iterArgs.get(i).name(), iterValues[i]);
                        }
                        List<ScalarValue> yielded = executeNode(loop.body(), iterEnv, bindings);
                        if (yielded == null) {
                            throw new IllegalStateException("Structured loop body must yield");
                        }
                        if (yielded.size() != iterArgs.size()) {
                            throw new IllegalStateException(
                                    "Yield arity "
                                            + yielded.size()
                                            + " does not match iter args "
                                            + iterArgs.size());
                        }
                        for (int i = 0; i < iterArgs.size(); i++) {
                            iterValues[i] = castValue(yielded.get(i), iterArgs.get(i).dataType());
                        }
                    }
                }
                for (int i = 0; i < iterArgs.size(); i++) {
                    env.scalarVars.put(iterArgs.get(i).name(), iterValues[i]);
                }
                return null;
            }
            case Yield yield -> {
                List<ScalarValue> values = new ArrayList<>(yield.values().size());
                for (LIRExprNode value : yield.values()) {
                    values.add(evalScalar(value, env, bindings));
                }
                return values;
            }
            default -> throw new IllegalStateException("Unexpected node: " + node.kind());
        }
    }

    private ScalarValue evalScalar(LIRExprNode node, Env env, Bindings bindings) {
        LIRExprNode resolved = bindings.graph.resolve(node);
        if (resolved.kind() != LIRExprKind.S_LOAD) {
            ScalarValue cached = env.scalarCache.get(resolved);
            if (cached != null) {
                return cached;
            }
        }
        ScalarValue value =
                switch (resolved.kind()) {
                    case S_CONST -> scalarConst((SConst) resolved);
                    case S_INPUT -> scalarInput((SInput) resolved, bindings);
                    case S_REF -> scalarRef((SRef) resolved, env);
                    case S_FROM_INDEX ->
                            castFromIndex(
                                    evalIndex(((SFromIndex) resolved).indexExpr(), env, bindings),
                                    resolved.dataType());
                    case S_LOAD -> scalarLoad((SLoad) resolved, env, bindings);
                    case S_UNARY -> evalUnary((SUnary) resolved, env, bindings);
                    case S_BINARY -> evalBinary((SBinary) resolved, env, bindings);
                    case S_TERNARY -> evalTernary((STernary) resolved, env, bindings);
                    case S_CAST -> evalCast((SCast) resolved, env, bindings);
                    default ->
                            throw new IllegalStateException(
                                    "Expected scalar node, got " + resolved.kind());
                };
        if (resolved.kind() != LIRExprKind.S_LOAD) {
            env.scalarCache.put(resolved, value);
        }
        return value;
    }

    private long evalIndex(LIRExprNode node, Env env, Bindings bindings) {
        LIRExprNode resolved = bindings.graph.resolve(node);
        Long cached = env.indexCache.get(resolved);
        if (cached != null) {
            return cached;
        }
        long value =
                switch (resolved.kind()) {
                    case I_CONST -> ((IConst) resolved).value();
                    case I_VAR -> {
                        String name = ((IVar) resolved).name();
                        Long idxValue = env.indexVars.get(name);
                        if (idxValue == null) {
                            throw new IllegalStateException("Unknown index variable: " + name);
                        }
                        yield idxValue;
                    }
                    case I_BINARY -> {
                        IBinary binary = (IBinary) resolved;
                        long left = evalIndex(binary.left(), env, bindings);
                        long right = evalIndex(binary.right(), env, bindings);
                        yield switch (binary.op()) {
                            case ADD -> left + right;
                            case SUBTRACT -> left - right;
                            case MULTIPLY -> left * right;
                            case DIVIDE -> left / right;
                            case MODULO -> left % right;
                            case BITWISE_AND -> left & right;
                            case SHIFT_LEFT -> left << right;
                            case SHIFT_RIGHT -> left >> right;
                        };
                    }
                    default ->
                            throw new IllegalStateException(
                                    "Expected index node, got " + resolved.kind());
                };
        env.indexCache.put(resolved, value);
        return value;
    }

    private ScalarValue scalarConst(SConst constant) {
        DataType type = constant.dataType();
        long rawBits = constant.rawBits();
        if (type == DataType.FP16) {
            float value = Float.float16ToFloat((short) rawBits);
            return ScalarValue.ofFloat(value, type);
        }
        if (type == DataType.BF16) {
            float value = BFloat16.toFloat((short) rawBits);
            return ScalarValue.ofFloat(value, type);
        }
        if (type == DataType.FP32) {
            return ScalarValue.ofFloat(Float.intBitsToFloat((int) rawBits), type);
        }
        if (type == DataType.FP64) {
            return ScalarValue.ofDouble(Double.longBitsToDouble(rawBits), type);
        }
        return ScalarValue.ofLong(rawBits, type);
    }

    private ScalarValue scalarInput(SInput input, Bindings bindings) {
        ScalarValue value = bindings.scalarInputs.get(input.inputId());
        if (value == null) {
            throw new IllegalStateException("Unknown scalar input id: " + input.inputId());
        }
        return value;
    }

    private ScalarValue scalarRef(SRef ref, Env env) {
        ScalarValue value = env.scalarVars.get(ref.name());
        if (value == null) {
            throw new IllegalStateException("Unknown scalar ref: " + ref.name());
        }
        return value;
    }

    private ScalarValue scalarLoad(SLoad load, Env env, Bindings bindings) {
        long offset = evalIndex(load.offset(), env, bindings);
        MemoryView<?> view = bindings.buffer(load.buffer());
        long address = view.byteOffset() + offset;
        DataType type = load.dataType();
        return readScalar(view, type, address, bindings.access);
    }

    private ScalarValue evalUnary(SUnary unary, Env env, Bindings bindings) {
        ScalarValue input = evalScalar(unary.input(), env, bindings);
        DataType type = unary.dataType();
        return switch (unary.op()) {
            case NEGATE -> castFromDouble(-toDouble(input), type);
            case ABS -> {
                if (type.isIntegral() || type == DataType.BOOL) {
                    long value = Math.abs(toLong(input));
                    yield castFromLong(value, type);
                }
                yield castFromDouble(Math.abs(toDouble(input)), type);
            }
            case EXP -> castFromDouble(Math.exp(toDouble(input)), type);
            case LOG -> castFromDouble(Math.log(toDouble(input)), type);
            case SQRT -> castFromDouble(Math.sqrt(toDouble(input)), type);
            case SQUARE -> {
                double value = toDouble(input);
                yield castFromDouble(value * value, type);
            }
            case SIN -> castFromDouble(Math.sin(toDouble(input)), type);
            case COS -> castFromDouble(Math.cos(toDouble(input)), type);
            case TAN -> castFromDouble(Math.tan(toDouble(input)), type);
            case TANH -> castFromDouble(Math.tanh(toDouble(input)), type);
            case RECIPROCAL -> castFromDouble(1.0 / toDouble(input), type);
            case LOGICAL_NOT -> ScalarValue.ofBool(!toBoolean(input), DataType.BOOL);
            case BITWISE_NOT -> castFromLong(~toLong(input), type);
        };
    }

    private ScalarValue evalBinary(SBinary binary, Env env, Bindings bindings) {
        ScalarValue left = evalScalar(binary.left(), env, bindings);
        ScalarValue right = evalScalar(binary.right(), env, bindings);
        DataType type = binary.dataType();
        return switch (binary.op()) {
            case ADD -> castFromBinary(left, right, type, (l, r) -> l + r, (l, r) -> l + r);
            case SUBTRACT -> castFromBinary(left, right, type, (l, r) -> l - r, (l, r) -> l - r);
            case MULTIPLY -> castFromBinary(left, right, type, (l, r) -> l * r, (l, r) -> l * r);
            case DIVIDE -> castFromBinary(left, right, type, (l, r) -> l / r, (l, r) -> l / r);
            case MIN -> castFromBinary(left, right, type, Math::min, (l, r) -> l < r ? l : r);
            case MAX -> castFromBinary(left, right, type, Math::max, (l, r) -> l > r ? l : r);
            case POW -> castFromDouble(Math.pow(toDouble(left), toDouble(right)), type);
            case LOGICAL_AND -> castFromBool(toBoolean(left) && toBoolean(right), type);
            case LOGICAL_OR -> castFromBool(toBoolean(left) || toBoolean(right), type);
            case LOGICAL_XOR -> castFromBool(toBoolean(left) ^ toBoolean(right), type);
            case BITWISE_AND -> castFromLong(toLong(left) & toLong(right), type);
            case BITWISE_OR -> castFromLong(toLong(left) | toLong(right), type);
            case BITWISE_XOR -> castFromLong(toLong(left) ^ toLong(right), type);
            case EQUAL -> castFromBool(compareEqual(left, right), type);
            case LESS_THAN -> castFromBool(compareLessThan(left, right), type);
        };
    }

    private ScalarValue evalTernary(STernary ternary, Env env, Bindings bindings) {
        ScalarValue cond = evalScalar(ternary.condition(), env, bindings);
        boolean chooseTrue = toBoolean(cond);
        ScalarValue chosen =
                chooseTrue
                        ? evalScalar(ternary.trueValue(), env, bindings)
                        : evalScalar(ternary.falseValue(), env, bindings);
        return castValue(chosen, ternary.dataType());
    }

    private ScalarValue evalCast(SCast cast, Env env, Bindings bindings) {
        ScalarValue input = evalScalar(cast.input(), env, bindings);
        return castValue(input, cast.targetType());
    }

    private boolean compareEqual(ScalarValue left, ScalarValue right) {
        if (left.type == DataType.BOOL && right.type == DataType.BOOL) {
            return toBoolean(left) == toBoolean(right);
        }
        if (left.type.isFloatingPoint() || right.type.isFloatingPoint()) {
            return Double.compare(toDouble(left), toDouble(right)) == 0;
        }
        return toLong(left) == toLong(right);
    }

    private boolean compareLessThan(ScalarValue left, ScalarValue right) {
        if (left.type == DataType.BOOL || right.type == DataType.BOOL) {
            return (int) toLong(left) < (int) toLong(right);
        }
        if (left.type.isFloatingPoint() || right.type.isFloatingPoint()) {
            return toDouble(left) < toDouble(right);
        }
        return toLong(left) < toLong(right);
    }

    private ScalarValue castFromIndex(long index, DataType target) {
        return castFromLong(index, target);
    }

    private ScalarValue castFromBool(boolean value, DataType target) {
        if (target == DataType.BOOL) {
            return ScalarValue.ofBool(value, DataType.BOOL);
        }
        return castFromLong(value ? 1 : 0, target);
    }

    private ScalarValue castFromLong(long value, DataType target) {
        if (target == DataType.BOOL) {
            return ScalarValue.ofBool(value != 0, DataType.BOOL);
        }
        if (target.isFloatingPoint()) {
            return castFromDouble(value, target);
        }
        return ScalarValue.ofLong(value, target);
    }

    private ScalarValue castFromDouble(double value, DataType target) {
        if (target == DataType.FP64) {
            return ScalarValue.ofDouble(value, target);
        }
        if (target == DataType.FP32 || target == DataType.FP16 || target == DataType.BF16) {
            return ScalarValue.ofFloat((float) value, target);
        }
        if (target == DataType.BOOL) {
            return ScalarValue.ofBool(value != 0.0, DataType.BOOL);
        }
        return ScalarValue.ofLong((long) value, target);
    }

    private ScalarValue castValue(ScalarValue value, DataType target) {
        if (value.type == target) {
            return value;
        }
        if (target.isFloatingPoint()) {
            return castFromDouble(toDouble(value), target);
        }
        return castFromLong(toLong(value), target);
    }

    private ScalarValue castFromBinary(
            ScalarValue left,
            ScalarValue right,
            DataType target,
            DoubleBinaryOp floatOp,
            LongBinaryOp intOp) {
        if (target.isFloatingPoint()) {
            double result = floatOp.apply(toDouble(left), toDouble(right));
            return castFromDouble(result, target);
        }
        long result = intOp.apply(toLong(left), toLong(right));
        return castFromLong(result, target);
    }

    private double toDouble(ScalarValue value) {
        if (value.type == DataType.FP64) {
            return Double.longBitsToDouble(value.bits);
        }
        if (value.type == DataType.FP32
                || value.type == DataType.FP16
                || value.type == DataType.BF16) {
            return Float.intBitsToFloat((int) value.bits);
        }
        if (value.type == DataType.BOOL) {
            return value.bits != 0 ? 1.0 : 0.0;
        }
        if (value.type == DataType.I8) {
            return (byte) value.bits;
        }
        if (value.type == DataType.I16) {
            return (short) value.bits;
        }
        if (value.type == DataType.I32) {
            return (int) value.bits;
        }
        return value.bits;
    }

    private long toLong(ScalarValue value) {
        if (value.type == DataType.BOOL) {
            return value.bits != 0 ? 1 : 0;
        }
        if (value.type == DataType.I8) {
            return (byte) value.bits;
        }
        if (value.type == DataType.I16) {
            return (short) value.bits;
        }
        if (value.type == DataType.I32) {
            return (int) value.bits;
        }
        if (value.type == DataType.I64) {
            return value.bits;
        }
        return (long) toDouble(value);
    }

    private boolean toBoolean(ScalarValue value) {
        if (value.type == DataType.BOOL) {
            return value.bits != 0;
        }
        if (value.type.isFloatingPoint()) {
            return toDouble(value) != 0.0;
        }
        return toLong(value) != 0;
    }

    private ScalarValue readScalar(
            MemoryView<?> view, DataType type, long address, MemoryAccess<?> access) {
        MemoryView<Object> typedView = castView(view);
        MemoryAccess<Object> typedAccess = castAccess(access);
        if (type == DataType.FP32) {
            return ScalarValue.ofFloat(typedAccess.readFloat(typedView.memory(), address), type);
        }
        if (type == DataType.FP64) {
            return ScalarValue.ofDouble(typedAccess.readDouble(typedView.memory(), address), type);
        }
        if (type == DataType.FP16) {
            short bits = typedAccess.readShort(typedView.memory(), address);
            return ScalarValue.ofFloat(Float.float16ToFloat(bits), type);
        }
        if (type == DataType.BF16) {
            short bits = typedAccess.readShort(typedView.memory(), address);
            return ScalarValue.ofFloat(BFloat16.toFloat(bits), type);
        }
        if (type == DataType.I8 || type == DataType.BOOL) {
            return ScalarValue.ofLong(typedAccess.readByte(typedView.memory(), address), type);
        }
        if (type == DataType.I16) {
            return ScalarValue.ofLong(typedAccess.readShort(typedView.memory(), address), type);
        }
        if (type == DataType.I32) {
            return ScalarValue.ofLong(typedAccess.readInt(typedView.memory(), address), type);
        }
        if (type == DataType.I64) {
            return ScalarValue.ofLong(typedAccess.readLong(typedView.memory(), address), type);
        }
        throw new UnsupportedOperationException("Unsupported dtype: " + type);
    }

    private void writeScalar(
            MemoryView<?> view,
            DataType targetType,
            ScalarValue value,
            long offset,
            MemoryAccess<?> access) {
        long address = view.byteOffset() + offset;
        MemoryView<Object> typedView = castView(view);
        MemoryAccess<Object> typedAccess = castAccess(access);
        ScalarValue casted = castValue(value, targetType);
        if (targetType == DataType.FP32) {
            typedAccess.writeFloat(
                    typedView.memory(), address, Float.intBitsToFloat((int) casted.bits));
            return;
        }
        if (targetType == DataType.FP64) {
            typedAccess.writeDouble(
                    typedView.memory(), address, Double.longBitsToDouble(casted.bits));
            return;
        }
        if (targetType == DataType.FP16) {
            float v = Float.intBitsToFloat((int) casted.bits);
            typedAccess.writeShort(typedView.memory(), address, Float.floatToFloat16(v));
            return;
        }
        if (targetType == DataType.BF16) {
            float v = Float.intBitsToFloat((int) casted.bits);
            typedAccess.writeShort(typedView.memory(), address, BFloat16.fromFloat(v));
            return;
        }
        if (targetType == DataType.I8 || targetType == DataType.BOOL) {
            typedAccess.writeByte(typedView.memory(), address, (byte) casted.bits);
            return;
        }
        if (targetType == DataType.I16) {
            typedAccess.writeShort(typedView.memory(), address, (short) casted.bits);
            return;
        }
        if (targetType == DataType.I32) {
            typedAccess.writeInt(typedView.memory(), address, (int) casted.bits);
            return;
        }
        if (targetType == DataType.I64) {
            typedAccess.writeLong(typedView.memory(), address, casted.bits);
            return;
        }
        throw new UnsupportedOperationException("Unsupported dtype: " + targetType);
    }

    @SuppressWarnings("unchecked")
    private MemoryView<Object> castView(MemoryView<?> view) {
        return (MemoryView<Object>) view;
    }

    @SuppressWarnings("unchecked")
    private MemoryAccess<Object> castAccess(MemoryAccess<?> access) {
        return (MemoryAccess<Object>) access;
    }

    private record ScalarValue(DataType type, long bits) {
        static ScalarValue ofFloat(float value, DataType type) {
            return new ScalarValue(type, Float.floatToRawIntBits(value));
        }

        static ScalarValue ofDouble(double value, DataType type) {
            return new ScalarValue(type, Double.doubleToRawLongBits(value));
        }

        static ScalarValue ofLong(long value, DataType type) {
            return new ScalarValue(type, value);
        }

        static ScalarValue ofBool(boolean value, DataType type) {
            return new ScalarValue(type, value ? 1L : 0L);
        }
    }

    private static final class Env {
        private final Map<String, Long> indexVars = new HashMap<>();
        private final Map<String, ScalarValue> scalarVars = new HashMap<>();
        private final Map<LIRExprNode, ScalarValue> scalarCache = new IdentityHashMap<>();
        private final Map<LIRExprNode, Long> indexCache = new IdentityHashMap<>();

        Env copy() {
            Env env = new Env();
            env.indexVars.putAll(indexVars);
            env.scalarVars.putAll(scalarVars);
            return env;
        }
    }

    private record Bindings(
            LIRExprGraph graph,
            MemoryAccess<?> access,
            Map<BufferRef, MemoryView<?>> buffers,
            Map<Integer, ScalarValue> scalarInputs) {

        static Bindings from(
                LIRGraph graph,
                List<MemoryView<?>> buffers,
                List<ScalarArg> scalars,
                List<MemoryView<?>> outputs,
                MemoryDomain<?> memoryDomain) {
            Map<BufferRef, MemoryView<?>> bufferMap = new HashMap<>();
            Map<Integer, ScalarValue> scalarMap = new HashMap<>();
            int bufferIndex = 0;
            int scalarIndex = 0;

            for (LIRInput input : graph.inputs()) {
                if (input instanceof ScalarInput scalarInput) {
                    if (scalarIndex >= scalars.size()) {
                        throw new IllegalArgumentException(
                                "Missing scalar input at index " + scalarIndex);
                    }
                    ScalarArg scalar = scalars.get(scalarIndex++);
                    if (scalar.dataType() != scalarInput.dataType()) {
                        throw new IllegalArgumentException(
                                "Scalar input type mismatch: expected "
                                        + scalarInput.dataType()
                                        + " but got "
                                        + scalar.dataType());
                    }
                    scalarMap.put(scalarInput.id(), fromScalarArg(scalar, scalarInput.dataType()));
                } else {
                    if (bufferIndex >= buffers.size()) {
                        throw new IllegalArgumentException(
                                "Missing buffer input at index " + bufferIndex);
                    }
                    bufferMap.put((BufferRef) input, buffers.get(bufferIndex++));
                }
            }

            if (bufferIndex != buffers.size()) {
                throw new IllegalArgumentException(
                        "Expected " + bufferIndex + " buffer inputs but got " + buffers.size());
            }
            if (scalarIndex != scalars.size()) {
                throw new IllegalArgumentException(
                        "Expected " + scalarIndex + " scalar inputs but got " + scalars.size());
            }

            if (outputs.size() != graph.outputs().size()) {
                throw new IllegalArgumentException(
                        "Expected "
                                + graph.outputs().size()
                                + " outputs but got "
                                + outputs.size());
            }
            for (int i = 0; i < graph.outputs().size(); i++) {
                bufferMap.put(graph.outputs().get(i), outputs.get(i));
            }

            return new Bindings(
                    graph.exprGraph(), memoryDomain.directAccess(), bufferMap, scalarMap);
        }

        MemoryView<?> buffer(BufferRef ref) {
            MemoryView<?> view = buffers.get(ref);
            if (view == null) {
                throw new IllegalStateException("Missing buffer binding for " + ref);
            }
            return view;
        }

        private static ScalarValue fromScalarArg(ScalarArg arg, DataType dataType) {
            long rawBits = arg.rawBits();
            if (dataType == DataType.FP16) {
                float value = Float.float16ToFloat((short) rawBits);
                return ScalarValue.ofFloat(value, dataType);
            }
            if (dataType == DataType.BF16) {
                float value = BFloat16.toFloat((short) rawBits);
                return ScalarValue.ofFloat(value, dataType);
            }
            if (dataType == DataType.FP32) {
                return ScalarValue.ofFloat(Float.intBitsToFloat((int) rawBits), dataType);
            }
            if (dataType == DataType.FP64) {
                return ScalarValue.ofDouble(Double.longBitsToDouble(rawBits), dataType);
            }
            return ScalarValue.ofLong(rawBits, dataType);
        }
    }

    private interface DoubleBinaryOp {
        double apply(double left, double right);
    }

    private interface LongBinaryOp {
        long apply(long left, long right);
    }
}

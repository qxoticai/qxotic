package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;

/** Interpreter for IR-L programs. Executes IR-L graphs directly for testing and verification. */
public final class LIRInterpreter {

    private final Map<Integer, MemorySegment> buffers = new HashMap<>();
    private final Map<Integer, ScalarInputValue> scalarInputs = new HashMap<>();
    private final Map<String, Long> indexVars = new HashMap<>();
    private final Map<String, ScalarValue> scalarBindings = new HashMap<>();

    private record ScalarInputValue(long rawBits, DataType dataType) {}

    private record ScalarValue(long rawBits, DataType dataType) {}

    /** Binds a buffer to the given id. */
    public void bindBuffer(int id, MemorySegment segment) {
        buffers.put(id, segment);
    }

    /** Binds a float input value to the given id. */
    public void bindFloatInput(int id, float value) {
        scalarInputs.put(id, new ScalarInputValue(Float.floatToRawIntBits(value), DataType.FP32));
    }

    /** Binds a scalar input value (as raw bits) with its data type to the given id. */
    public void bindScalarInput(int id, long rawBits, DataType dataType) {
        scalarInputs.put(id, new ScalarInputValue(rawBits, dataType));
    }

    /** Executes an IR-L graph. */
    public void execute(LIRGraph graph) {
        executeNode(graph.body());
    }

    /** Executes a single IR-L node. */
    public void executeNode(LIRNode node) {
        switch (node) {
            case Loop loop -> executeLoop(loop);
            case StructuredFor structuredFor -> executeStructuredFor(structuredFor);
            case TiledLoop tiledLoop -> executeTiledLoop(tiledLoop);
            case LoopNest loopNest -> executeLoopNest(loopNest);
            case Block block -> executeBlock(block);
            case Store store -> executeStore(store);
            case ScalarLet let -> executeScalarLet(let);
            case Yield ignored -> {}
            case IndexExpr ignored -> {}
            case ScalarExpr ignored -> {}
            case BufferRef ignored -> {}
            case Load ignored -> {}
        }
    }

    private void executeScalarLet(ScalarLet let) {
        // Evaluate the value and bind it (available for rest of block)
        long valueBits = evaluateScalar(let.value());
        DataType dtype = let.value().dataType();
        scalarBindings.put(let.name(), new ScalarValue(valueBits, dtype));
    }

    private void executeLoop(Loop loop) {
        long bound = evaluateIndex(loop.bound());
        for (long i = 0; i < bound; i++) {
            indexVars.put(loop.indexName(), i);
            executeNode(loop.body());
        }
        indexVars.remove(loop.indexName());
    }

    private void executeStructuredFor(StructuredFor loop) {
        long lower = evaluateIndex(loop.lowerBound());
        long upper = evaluateIndex(loop.upperBound());
        long step = evaluateIndex(loop.step());
        if (step == 0) {
            throw new IllegalArgumentException("Loop step cannot be zero");
        }

        long[] iterValues = new long[loop.iterArgs().size()];
        DataType[] iterTypes = new DataType[loop.iterArgs().size()];
        for (int i = 0; i < loop.iterArgs().size(); i++) {
            LoopIterArg arg = loop.iterArgs().get(i);
            iterValues[i] = evaluateScalar(arg.init());
            iterTypes[i] = arg.dataType();
            scalarBindings.put(arg.name(), new ScalarValue(iterValues[i], iterTypes[i]));
        }

        for (long i = lower; step > 0 ? i < upper : i > upper; i += step) {
            indexVars.put(loop.indexName(), i);
            for (int j = 0; j < loop.iterArgs().size(); j++) {
                LoopIterArg arg = loop.iterArgs().get(j);
                scalarBindings.put(arg.name(), new ScalarValue(iterValues[j], iterTypes[j]));
            }
            long[] yielded = executeBodyAndYield(loop.body(), loop.iterArgs().size());
            System.arraycopy(yielded, 0, iterValues, 0, yielded.length);
        }

        indexVars.remove(loop.indexName());
        for (int i = 0; i < loop.iterArgs().size(); i++) {
            LoopIterArg arg = loop.iterArgs().get(i);
            scalarBindings.put(arg.name(), new ScalarValue(iterValues[i], iterTypes[i]));
        }
    }

    private long[] executeBodyAndYield(LIRNode body, int expectedArity) {
        if (body instanceof Yield yield) {
            return evaluateYield(yield, expectedArity);
        }
        if (body instanceof Block block) {
            if (block.statements().isEmpty()) {
                throw new IllegalStateException("Structured loop body cannot be empty");
            }
            for (int i = 0; i < block.statements().size() - 1; i++) {
                executeNode(block.statements().get(i));
            }
            LIRNode last = block.statements().getLast();
            if (!(last instanceof Yield yield)) {
                throw new IllegalStateException("Structured loop body must end with Yield");
            }
            return evaluateYield(yield, expectedArity);
        }
        throw new IllegalStateException("Structured loop body must end with Yield");
    }

    private long[] evaluateYield(Yield yield, int expectedArity) {
        if (yield.values().size() != expectedArity) {
            throw new IllegalStateException(
                    "Yield arity "
                            + yield.values().size()
                            + " does not match iter args "
                            + expectedArity);
        }
        long[] values = new long[yield.values().size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = evaluateScalar(yield.values().get(i));
        }
        return values;
    }

    private void executeTiledLoop(TiledLoop tiledLoop) {
        long totalBound = evaluateIndex(tiledLoop.totalBound());
        long tileSize = tiledLoop.tileSize();
        long numTiles = (totalBound + tileSize - 1) / tileSize;

        for (long tile = 0; tile < numTiles; tile++) {
            indexVars.put(tiledLoop.outerName(), tile);
            long start = tile * tileSize;
            long end = Math.min(start + tileSize, totalBound);
            for (long i = start; i < end; i++) {
                indexVars.put(tiledLoop.innerName(), i);
                executeNode(tiledLoop.body());
            }
        }
        indexVars.remove(tiledLoop.outerName());
        indexVars.remove(tiledLoop.innerName());
    }

    private void executeLoopNest(LoopNest loopNest) {
        executeNestedLoops(loopNest.loops(), 0, loopNest.body());
    }

    private void executeNestedLoops(java.util.List<Loop> loops, int depth, LIRNode body) {
        if (depth >= loops.size()) {
            executeNode(body);
            return;
        }

        Loop loop = loops.get(depth);
        long bound = evaluateIndex(loop.bound());
        for (long i = 0; i < bound; i++) {
            indexVars.put(loop.indexName(), i);
            executeNestedLoops(loops, depth + 1, body);
        }
        indexVars.remove(loop.indexName());
    }

    private void executeBlock(Block block) {
        for (LIRNode stmt : block.statements()) {
            executeNode(stmt);
        }
    }

    private void executeStore(Store store) {
        MemorySegment buffer = buffers.get(store.buffer().id());
        long offset = evaluateIndex(store.offset());
        long valueBits = evaluateScalar(store.value());
        writeValue(buffer, offset, valueBits, store.buffer().dataType());
    }

    /** Evaluates an index expression and returns its value. */
    public long evaluateIndex(IndexExpr expr) {
        return switch (expr) {
            case IndexConst c -> c.value();
            case IndexVar v -> {
                Long val = indexVars.get(v.name());
                if (val == null) {
                    throw new IllegalStateException("Index variable not bound: " + v.name());
                }
                yield val;
            }
            case IndexBinary b -> {
                long left = evaluateIndex(b.left());
                long right = evaluateIndex(b.right());
                yield switch (b.op()) {
                    case ADD -> left + right;
                    case SUBTRACT -> left - right;
                    case MULTIPLY -> left * right;
                    case DIVIDE -> left / right;
                    case MODULO -> left % right;
                    case BITWISE_AND -> left & right;
                    case SHIFT_LEFT -> left << right;
                    case SHIFT_RIGHT -> left >>> right; // Unsigned right shift
                };
            }
        };
    }

    /** Evaluates a scalar expression and returns its raw bits. */
    public long evaluateScalar(ScalarExpr expr) {
        return switch (expr) {
            case ScalarLiteral c -> c.rawBits();
            case ScalarLoad load -> {
                MemorySegment buffer = buffers.get(load.buffer().id());
                long offset = evaluateIndex(load.offset());
                yield readValue(buffer, offset, load.buffer().dataType());
            }
            case ScalarInput input -> {
                // Scalar input: read from bound scalar value
                ScalarInputValue value = scalarInputs.get(input.id());
                if (value == null) {
                    throw new IllegalStateException("Scalar input not bound: " + input.id());
                }
                yield value.rawBits();
            }
            case ScalarUnary u -> evaluateUnary(u);
            case ScalarBinary b -> evaluateBinary(b);
            case ScalarTernary t -> evaluateTernary(t);
            case ScalarCast c -> evaluateCast(c);
            case ScalarFromIndex sfi -> evaluateIndex(sfi.index());
            case ScalarRef ref -> {
                // Reference to a let-bound scalar
                ScalarValue value = scalarBindings.get(ref.name());
                if (value == null) {
                    throw new IllegalStateException("Scalar binding not found: " + ref.name());
                }
                yield value.rawBits();
            }
        };
    }

    private long evaluateUnary(ScalarUnary node) {
        long inputBits = evaluateScalar(node.input());
        DataType dtype = node.input().dataType();
        UnaryOperator op = node.op();

        if (dtype == DataType.FP32) {
            float a = Float.intBitsToFloat((int) inputBits);
            float result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> (float) Math.exp(a);
                        case LOG -> (float) Math.log(a);
                        case SQRT -> (float) Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> (float) Math.sin(a);
                        case COS -> (float) Math.cos(a);
                        case TAN -> (float) Math.tan(a);
                        case TANH -> (float) Math.tanh(a);
                        case RECIPROCAL -> 1.0f / a;
                        case LOGICAL_NOT, BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for FP32");
                    };
            return Float.floatToRawIntBits(result);
        } else if (dtype == DataType.FP64) {
            double a = Double.longBitsToDouble(inputBits);
            double result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> Math.exp(a);
                        case LOG -> Math.log(a);
                        case SQRT -> Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> Math.sin(a);
                        case COS -> Math.cos(a);
                        case TAN -> Math.tan(a);
                        case TANH -> Math.tanh(a);
                        case RECIPROCAL -> 1.0 / a;
                        case LOGICAL_NOT, BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for FP64");
                    };
            return Double.doubleToRawLongBits(result);
        } else if (dtype == DataType.I32) {
            int a = (int) inputBits;
            int result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case BITWISE_NOT -> ~a;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for I32");
                    };
            return result;
        } else if (dtype == DataType.I64) {
            long result =
                    switch (op) {
                        case NEGATE -> -inputBits;
                        case ABS -> Math.abs(inputBits);
                        case BITWISE_NOT -> ~inputBits;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for I64");
                    };
            return result;
        } else if (dtype == DataType.BOOL) {
            boolean a = inputBits != 0;
            boolean result =
                    switch (op) {
                        case LOGICAL_NOT -> !a;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for BOOL");
                    };
            return result ? 1 : 0;
        }
        throw new UnsupportedOperationException("Unsupported dtype for unary: " + dtype);
    }

    private long evaluateBinary(ScalarBinary node) {
        long leftBits = evaluateScalar(node.left());
        long rightBits = evaluateScalar(node.right());
        DataType dtype = node.left().dataType();
        BinaryOperator op = node.op();

        if (dtype == DataType.FP32) {
            float a = Float.intBitsToFloat((int) leftBits);
            float b = Float.intBitsToFloat((int) rightBits);
            float result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> (float) Math.pow(a, b);
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for FP32");
                    };
            return Float.floatToRawIntBits(result);
        } else if (dtype == DataType.FP64) {
            double a = Double.longBitsToDouble(leftBits);
            double b = Double.longBitsToDouble(rightBits);
            double result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> Math.pow(a, b);
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for FP64");
                    };
            return Double.doubleToRawLongBits(result);
        } else if (dtype == DataType.I32) {
            int a = (int) leftBits;
            int b = (int) rightBits;
            int result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case BITWISE_AND -> a & b;
                        case BITWISE_OR -> a | b;
                        case BITWISE_XOR -> a ^ b;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for I32");
                    };
            return result;
        } else if (dtype == DataType.I64) {
            long result =
                    switch (op) {
                        case ADD -> leftBits + rightBits;
                        case SUBTRACT -> leftBits - rightBits;
                        case MULTIPLY -> leftBits * rightBits;
                        case DIVIDE -> leftBits / rightBits;
                        case MIN -> Math.min(leftBits, rightBits);
                        case MAX -> Math.max(leftBits, rightBits);
                        case BITWISE_AND -> leftBits & rightBits;
                        case BITWISE_OR -> leftBits | rightBits;
                        case BITWISE_XOR -> leftBits ^ rightBits;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for I64");
                    };
            return result;
        } else if (dtype == DataType.BOOL) {
            boolean a = leftBits != 0;
            boolean b = rightBits != 0;
            boolean result =
                    switch (op) {
                        case LOGICAL_AND -> a && b;
                        case LOGICAL_OR -> a || b;
                        case LOGICAL_XOR -> a ^ b;
                        default ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for BOOL");
                    };
            return result ? 1 : 0;
        }
        throw new UnsupportedOperationException("Unsupported dtype for binary: " + dtype);
    }

    private long evaluateTernary(ScalarTernary node) {
        long condBits = evaluateScalar(node.condition());
        boolean cond = condBits != 0;
        return cond ? evaluateScalar(node.trueValue()) : evaluateScalar(node.falseValue());
    }

    private long evaluateCast(ScalarCast node) {
        long inputBits = evaluateScalar(node.input());
        DataType srcType = node.input().dataType();
        DataType dstType = node.targetType();

        double value = toDouble(inputBits, srcType);
        return fromDouble(value, dstType);
    }

    private double toDouble(long bits, DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.intBitsToFloat((int) bits);
        } else if (dtype == DataType.FP64) {
            return Double.longBitsToDouble(bits);
        } else if (dtype == DataType.FP16) {
            return Float.float16ToFloat((short) bits);
        } else if (dtype == DataType.BF16) {
            return BFloat16.toFloat((short) bits);
        } else if (dtype == DataType.I8) {
            return (byte) bits;
        } else if (dtype == DataType.I16) {
            return (short) bits;
        } else if (dtype == DataType.I32) {
            return (int) bits;
        } else if (dtype == DataType.I64) {
            return bits;
        } else if (dtype == DataType.BOOL) {
            return bits != 0 ? 1.0 : 0.0;
        }
        throw new UnsupportedOperationException("Cannot convert to double from: " + dtype);
    }

    private long fromDouble(double value, DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits((float) value);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(value);
        } else if (dtype == DataType.FP16) {
            return Float.floatToFloat16((float) value);
        } else if (dtype == DataType.BF16) {
            return BFloat16.fromFloat((float) value);
        } else if (dtype == DataType.I8) {
            return (byte) value;
        } else if (dtype == DataType.I16) {
            return (short) value;
        } else if (dtype == DataType.I32) {
            return (int) value;
        } else if (dtype == DataType.I64) {
            return (long) value;
        } else if (dtype == DataType.BOOL) {
            return value != 0 ? 1 : 0;
        }
        throw new UnsupportedOperationException("Cannot convert from double to: " + dtype);
    }

    private long readValue(MemorySegment segment, long offset, DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset));
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(
                    segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset));
        } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
            return segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        } else if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            return segment.get(ValueLayout.JAVA_BYTE, offset);
        } else if (dtype == DataType.I16) {
            return segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        } else if (dtype == DataType.I32) {
            return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        } else if (dtype == DataType.I64) {
            return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        }
        throw new UnsupportedOperationException("Unsupported dtype for read: " + dtype);
    }

    private void writeValue(MemorySegment segment, long offset, long rawBits, DataType dtype) {
        if (dtype == DataType.FP32) {
            segment.set(
                    ValueLayout.JAVA_FLOAT_UNALIGNED, offset, Float.intBitsToFloat((int) rawBits));
        } else if (dtype == DataType.FP64) {
            segment.set(
                    ValueLayout.JAVA_DOUBLE_UNALIGNED, offset, Double.longBitsToDouble(rawBits));
        } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
            segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, (short) rawBits);
        } else if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            segment.set(ValueLayout.JAVA_BYTE, offset, (byte) rawBits);
        } else if (dtype == DataType.I16) {
            segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, (short) rawBits);
        } else if (dtype == DataType.I32) {
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, (int) rawBits);
        } else if (dtype == DataType.I64) {
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, rawBits);
        } else {
            throw new UnsupportedOperationException("Unsupported dtype for write: " + dtype);
        }
    }

    /** Clears all interpreter state. */
    public void reset() {
        buffers.clear();
        scalarInputs.clear();
        indexVars.clear();
        scalarBindings.clear();
    }
}

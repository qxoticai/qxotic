package com.qxotic.jota.stablehlo.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Util;
import com.qxotic.jota.ir.tir.BinaryOp;
import com.qxotic.jota.ir.tir.CastOp;
import com.qxotic.jota.ir.tir.Contiguous;
import com.qxotic.jota.ir.tir.GatherOp;
import com.qxotic.jota.ir.tir.IotaConstant;
import com.qxotic.jota.ir.tir.KernelStep;
import com.qxotic.jota.ir.tir.RandomUniformOp;
import com.qxotic.jota.ir.tir.ReductionOp;
import com.qxotic.jota.ir.tir.ReductionOperator;
import com.qxotic.jota.ir.tir.ScalarConstant;
import com.qxotic.jota.ir.tir.ScalarInput;
import com.qxotic.jota.ir.tir.ScheduleInputRef;
import com.qxotic.jota.ir.tir.ScheduledOutputRef;
import com.qxotic.jota.ir.tir.ScheduledProgram;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TIRVisitor;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.ir.tir.TernaryOp;
import com.qxotic.jota.ir.tir.UnaryOp;
import com.qxotic.jota.ir.tir.ValueId;
import com.qxotic.jota.ir.tir.ViewOperation;
import com.qxotic.jota.ir.tir.ViewTransform;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Exports a TIR graph as a StableHLO program: {@code func.func @main} whose arguments are the graph
 * inputs (in order) and whose results are the graph outputs.
 *
 * <p>The export preserves <b>TIR</b> semantics where TIR and StableHLO differ (see {@code
 * jota/TODO-parity-TIR-StableHLO.md}, section 4): implicit broadcasting becomes {@code
 * broadcast_in_dim}, shift amounts are masked to the bit width as TIR does, reductions get an
 * explicit identity init value and a reshape for {@code keepDims}, and {@link RandomUniformOp} is
 * expanded into the exact integer arithmetic of jota's counter-based generator. Layouts and {@link
 * Contiguous} disappear: StableHLO values have no layout.
 *
 * <p>Not supported: block-quantized data types (no StableHLO counterpart).
 */
public final class TirToStableHlo implements TIRVisitor<Value> {

    private final Block block = new Block();
    private final Map<TIRNode, Value> values = new IdentityHashMap<>();

    private TirToStableHlo() {}

    /** Exports {@code graph} as {@code func.func @main} in a new module. */
    public static Program export(TIRGraph graph) {
        return module(List.of(function(graph, "main", null)));
    }

    /**
     * Exports a scheduled program: one private function {@code @step_<k>} per kernel step, and a
     * {@code @main} calling them in order. {@code original} is the graph the schedule was derived
     * from; its inputs give the arguments of {@code @main} (input ids are positions in it).
     */
    public static Program export(ScheduledProgram schedule, TIRGraph original) {
        List<Operation> functions = new ArrayList<>();
        List<Operation> steps = new ArrayList<>();
        for (int k = 0; k < schedule.steps().size(); k++) {
            Operation f = function(schedule.steps().get(k).graph(), "step_" + k, "private");
            steps.add(f);
        }
        Block main = new Block();
        List<Value> args = new ArrayList<>();
        for (TIRNode input : original.inputs()) {
            args.add(main.addArgument(inputType(input), null));
        }
        Map<ValueId, Value> produced = new java.util.HashMap<>();
        for (int k = 0; k < schedule.steps().size(); k++) {
            KernelStep step = schedule.steps().get(k);
            List<Value> operands = new ArrayList<>();
            for (ScheduleInputRef ref : step.inputs()) {
                operands.add(
                        switch (ref) {
                            case ScheduleInputRef.TensorInputRef t -> args.get(t.inputId());
                            case ScheduleInputRef.ScalarInputRef sc -> args.get(sc.inputId());
                            case ScheduleInputRef.ProducedValueRef pv -> {
                                Value v = produced.get(pv.valueId());
                                if (v == null) {
                                    throw new IllegalArgumentException(
                                            "step "
                                                    + k
                                                    + " uses "
                                                    + pv.valueId()
                                                    + " before it is produced");
                                }
                                yield v;
                            }
                        });
            }
            FunctionType fn =
                    (FunctionType)
                            ((TypeAttr) steps.get(k).attribute("function_type").orElseThrow())
                                    .type();
            Operation call =
                    Operation.create(
                            "func.call",
                            operands,
                            fn.results(),
                            List.of(
                                    new NamedAttribute(
                                            "callee",
                                            com.qxotic.jota.stablehlo.attr.SymbolRefAttr.of(
                                                    "step_" + k))),
                            List.of(),
                            List.of());
            main.addOperation(call);
            produced.put(step.output(), call.result(0));
        }
        Value result =
                switch (schedule.output()) {
                    case ScheduledOutputRef.ValueOutput v -> produced.get(v.valueId());
                    case ScheduledOutputRef.TensorInputOutput t -> args.get(t.inputId());
                    case ScheduledOutputRef.ScalarInputOutput sc -> args.get(sc.inputId());
                };
        main.addOperation(op("func.return", List.of(result), List.of(), List.of()));
        functions.add(
                funcOp(
                        "main",
                        null,
                        new FunctionType(inputTypes(main), List.of(result.type())),
                        main));
        functions.addAll(steps);
        return module(functions);
    }

    private static Type inputType(TIRNode input) {
        return switch (input) {
            case TensorInput ti -> tensorType(ti.shape(), ti.dataType());
            case ScalarInput si -> tensorType(Shape.scalar(), si.dataType());
            default ->
                    throw new IllegalArgumentException(
                            "graph input is not an input node: " + input);
        };
    }

    /** {@code graph} as a function: arguments are its inputs, results its outputs. */
    private static Operation function(TIRGraph graph, String name, String visibility) {
        TirToStableHlo e = new TirToStableHlo();
        for (TIRNode input : graph.inputs()) {
            Value arg = e.block.addArgument(inputType(input), null);
            e.values.put(
                    input, input instanceof ScalarInput si ? e.broadcastTo(arg, si.shape()) : arg);
        }
        List<Value> outputs = new ArrayList<>();
        for (TIRNode output : graph.outputs()) {
            outputs.add(e.value(output));
        }
        e.block.addOperation(op("func.return", outputs, List.of(), List.of()));
        FunctionType fn =
                new FunctionType(inputTypes(e.block), outputs.stream().map(Value::type).toList());
        return funcOp(name, visibility, fn, e.block);
    }

    private static Operation funcOp(String name, String visibility, FunctionType fn, Block body) {
        List<NamedAttribute> props = new ArrayList<>();
        props.add(new NamedAttribute("sym_name", new StringAttr(name)));
        props.add(new NamedAttribute("function_type", new TypeAttr(fn)));
        if (visibility != null) {
            props.add(new NamedAttribute("sym_visibility", new StringAttr(visibility)));
        }
        return Operation.create(
                "func.func", List.of(), List.of(), props, List.of(), List.of(Region.of(body)));
    }

    private static Program module(List<Operation> functions) {
        Block moduleBody = new Block();
        functions.forEach(moduleBody::addOperation);
        return new Program(
                Operation.create(
                        "builtin.module",
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(Region.of(moduleBody))));
    }

    private static List<Type> inputTypes(Block b) {
        return b.arguments().stream().map(Value::type).toList();
    }

    private Value value(TIRNode node) {
        Value v = values.get(node);
        if (v == null) {
            v = node.accept(this);
            values.put(node, v);
        }
        return v;
    }

    // ================================================================== types

    static ElementType elementType(DataType dt) {
        if (dt == DataType.BOOL) {
            return IntegerType.I1;
        }
        if (dt == DataType.I8) {
            return IntegerType.I8;
        }
        if (dt == DataType.I16) {
            return IntegerType.I16;
        }
        if (dt == DataType.I32) {
            return IntegerType.I32;
        }
        if (dt == DataType.I64) {
            return IntegerType.I64;
        }
        if (dt == DataType.FP16) {
            return FloatType.F16;
        }
        if (dt == DataType.BF16) {
            return FloatType.BF16;
        }
        if (dt == DataType.FP32) {
            return FloatType.F32;
        }
        if (dt == DataType.FP64) {
            return FloatType.F64;
        }
        throw new IllegalArgumentException("no StableHLO element type for " + dt);
    }

    static TensorType tensorType(Shape shape, DataType dt) {
        return TensorType.of(elementType(dt), shape.flattenModes().toArray());
    }

    private static TensorType type(Value v) {
        return (TensorType) v.type();
    }

    // ================================================================== op building

    private static Operation op(
            String name, List<Value> operands, List<Type> results, List<NamedAttribute> props) {
        return Operation.create(name, operands, results, props, List.of(), List.of());
    }

    private Value emit(String name, List<Value> operands, Type result, NamedAttribute... props) {
        Operation o = op(name, operands, List.of(result), List.of(props));
        block.addOperation(o);
        return o.result(0);
    }

    private Value constant(TensorType type, long bits) {
        return emit(
                "stablehlo.constant",
                List.of(),
                type,
                new NamedAttribute("value", DenseElementsAttr.splatBits(type, bits)));
    }

    private Value scalar(DataType dt, double value) {
        return constant(tensorType(Shape.scalar(), dt), bitsOf(dt, value));
    }

    private static long bitsOf(DataType dt, double value) {
        if (dt.isFloatingPoint()) {
            return ((FloatType) elementType(dt)).format().encode(value);
        }
        return (long) value;
    }

    /** {@code broadcast_in_dim} with numpy (right-aligned) semantics, a no-op on equal shapes. */
    private Value broadcastTo(Value v, Shape shape) {
        TensorType from = type(v);
        long[] to = shape.flattenModes().toArray();
        if (Arrays.equals(from.dims(), to)) {
            return v;
        }
        int offset = to.length - from.rank();
        long[] dims = new long[from.rank()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = offset + i;
        }
        return emit(
                "stablehlo.broadcast_in_dim",
                List.of(v),
                from.withShape(to),
                new NamedAttribute("broadcast_dimensions", DenseArrayAttr.i64(dims)));
    }

    private Value reshape(Value v, long[] dims) {
        if (Arrays.equals(type(v).dims(), dims)) {
            return v;
        }
        return emit("stablehlo.reshape", List.of(v), type(v).withShape(dims));
    }

    private Value convert(Value v, ElementType to) {
        if (type(v).elementType().equals(to)) {
            return v;
        }
        return emit("stablehlo.convert", List.of(v), type(v).withElementType(to));
    }

    private Value binary(String name, Value a, Value b) {
        return emit("stablehlo." + name, List.of(a, b), type(a));
    }

    private Value compare(String direction, Value a, Value b) {
        ElementType et = type(a).elementType();
        String compareType =
                et instanceof FloatType
                        ? "FLOAT"
                        : et instanceof IntegerType it && it.isBoolean() ? "UNSIGNED" : "SIGNED";
        return emit(
                "stablehlo.compare",
                List.of(a, b),
                type(a).withElementType(IntegerType.I1),
                new NamedAttribute(
                        "comparison_direction",
                        EnumAttr.stablehlo("comparison_direction", direction)),
                new NamedAttribute(
                        "compare_type", EnumAttr.stablehlo("comparison_type", compareType)));
    }

    private Value select(Value pred, Value t, Value f) {
        return emit("stablehlo.select", List.of(pred, t, f), type(t));
    }

    // ================================================================== leaves

    @Override
    public Value visitTensorInput(TensorInput node) {
        throw new IllegalArgumentException("tensor input #" + node.id() + " is not a graph input");
    }

    @Override
    public Value visitScalarInput(ScalarInput node) {
        throw new IllegalArgumentException("scalar input #" + node.id() + " is not a graph input");
    }

    @Override
    public Value visitScalarConstant(ScalarConstant node) {
        TensorType t = tensorType(node.shape(), node.dataType());
        long bits = node.rawBits();
        if (node.dataType() == DataType.BOOL) {
            bits = bits != 0 ? 1 : 0;
        }
        return constant(t, bits);
    }

    @Override
    public Value visitIotaConstant(IotaConstant node) {
        // TIR iota: the row-major linear index over the shape
        long n = node.count();
        TensorType flat = TensorType.of(elementType(node.dataType()), n);
        Value iota =
                emit(
                        "stablehlo.iota",
                        List.of(),
                        flat,
                        new NamedAttribute("iota_dimension", IntegerAttr.i64(0)));
        return reshape(iota, node.shape().flattenModes().toArray());
    }

    @Override
    public Value visitRandomUniformOp(RandomUniformOp node) {
        // jota RandomAlgorithms: index-hashed LCG, reproduced with wrapping i64 arithmetic
        Shape shape = node.shape().flattenModes();
        long n = shape.size();
        TensorType i64 = TensorType.of(IntegerType.I64, shape.toArray());
        Value index =
                reshape(
                        emit(
                                "stablehlo.iota",
                                List.of(),
                                TensorType.of(IntegerType.I64, n),
                                new NamedAttribute("iota_dimension", IntegerAttr.i64(0))),
                        shape.toArray());
        long k0 = Math.floorMod((int) node.key0(), 1024);
        long k1 = Math.floorMod((int) node.key1(), 1024);
        Value seed = binary("add", index, constant(i64, k0 * 1009L + k1 * 9176L));
        Value state0 = nextState(seed, i64);
        if (node.dataType() == DataType.FP32) {
            Value u24 = modPositive(state0, 1L << 24, i64);
            Value f = convert(u24, FloatType.F32);
            return binary(
                    "multiply", f, constant(type(f), FloatFormat.F32.encode(Math.scalb(1.0, -24))));
        }
        Value state1 = nextState(state0, i64);
        Value hi26 = modPositive(state0, 1L << 26, i64);
        Value lo27 = modPositive(state1, 1L << 27, i64);
        Value bits53 = binary("add", binary("multiply", hi26, constant(i64, 1L << 27)), lo27);
        Value d = convert(bits53, FloatType.F64);
        return binary(
                "multiply", d, constant(type(d), FloatFormat.F64.encode(Math.scalb(1.0, -53))));
    }

    private Value nextState(Value v, TensorType i64) {
        return binary(
                "add", binary("multiply", v, constant(i64, 1664525L)), constant(i64, 1013904223L));
    }

    /** {@code ((v % bound) + bound) % bound}. */
    private Value modPositive(Value v, long bound, TensorType i64) {
        Value b = constant(i64, bound);
        return binary("remainder", binary("add", binary("remainder", v, b), b), b);
    }

    // ================================================================== elementwise

    @Override
    public Value visitUnaryOp(UnaryOp node) {
        Value x = value(node.input());
        String name =
                switch (node.op()) {
                    case NEGATE -> "negate";
                    case ABS -> "abs";
                    case EXP -> "exponential";
                    case LOG -> "log";
                    case SQRT -> "sqrt";
                    case SIN -> "sine";
                    case COS -> "cosine";
                    case TAN -> "tan";
                    case TANH -> "tanh";
                    case LOGICAL_NOT, BITWISE_NOT -> "not";
                    case RECIPROCAL -> null;
                };
        if (name == null) {
            Value one = broadcastTo(scalar(node.dataType(), 1.0), node.shape());
            return binary("divide", one, x);
        }
        return emit("stablehlo." + name, List.of(x), type(x));
    }

    @Override
    public Value visitBinaryOp(BinaryOp node) {
        Value a = broadcastTo(value(node.left()), node.shape());
        Value b = broadcastTo(value(node.right()), node.shape());
        ElementType et = type(a).elementType();
        b = convert(b, et);
        return switch (node.op()) {
            case ADD -> binary("add", a, b);
            case SUBTRACT -> binary("subtract", a, b);
            case MULTIPLY -> binary("multiply", a, b);
            case DIVIDE -> binary("divide", a, b);
            case MIN -> binary("minimum", a, b);
            case MAX -> binary("maximum", a, b);
            case POW -> binary("power", a, b);
            case LOGICAL_AND, BITWISE_AND -> binary("and", a, b);
            case LOGICAL_OR, BITWISE_OR -> binary("or", a, b);
            case LOGICAL_XOR, BITWISE_XOR -> binary("xor", a, b);
            case SHIFT_LEFT -> binary("shift_left", a, maskShift(b));
            case SHIFT_RIGHT -> binary("shift_right_arithmetic", a, maskShift(b));
            case SHIFT_RIGHT_UNSIGNED -> binary("shift_right_logical", a, maskShift(b));
            case EQUAL -> compare("EQ", a, b);
            case LESS_THAN -> compare("LT", a, b);
        };
    }

    /** TIR (Java) shifts use the amount modulo the bit width. */
    private Value maskShift(Value s) {
        int bits = ((IntegerType) type(s).elementType()).bitWidth();
        return binary("and", s, constant(type(s), bits - 1));
    }

    @Override
    public Value visitTernaryOp(TernaryOp node) {
        Value pred = broadcastTo(value(node.cond()), node.shape());
        Value t = broadcastTo(value(node.trueExpr()), node.shape());
        Value f = broadcastTo(value(node.falseExpr()), node.shape());
        return select(pred, t, f);
    }

    @Override
    public Value visitCastOp(CastOp node) {
        Value x = value(node.input());
        ElementType to = elementType(node.targetDataType());
        return convert(x, to);
    }

    // ================================================================== reductions & gather

    @Override
    public Value visitReductionOp(ReductionOp node) {
        Value x = value(node.input());
        DataType acc = node.accumulatorType();
        x = convert(x, elementType(acc));
        TensorType in = type(x);
        int rank = in.rank();
        boolean[] reduced = new boolean[rank];
        for (int axis : node.axes()) {
            reduced[Util.wrapAround(axis, rank)] = true;
        }
        List<Long> dims = new ArrayList<>();
        List<Long> kept = new ArrayList<>();
        for (int i = 0; i < rank; i++) {
            if (reduced[i]) {
                dims.add((long) i);
            } else {
                kept.add(in.dim(i));
            }
        }
        TensorType scalarType = TensorType.of(in.elementType());
        Value init = constant(scalarType, identity(node.op(), acc));
        String combiner =
                switch (node.op()) {
                    case SUM -> acc == DataType.BOOL ? "or" : "add";
                    case PROD -> acc == DataType.BOOL ? "and" : "multiply";
                    case MAX -> "maximum";
                    case MIN -> "minimum";
                };
        Block body = new Block();
        Value l = body.addArgument(scalarType, null);
        Value r = body.addArgument(scalarType, null);
        Operation c = op("stablehlo." + combiner, List.of(l, r), List.of(scalarType), List.of());
        body.addOperation(c);
        body.addOperation(op("stablehlo.return", List.of(c.result(0)), List.of(), List.of()));
        Operation reduce =
                Operation.create(
                        "stablehlo.reduce",
                        List.of(x, init),
                        List.of(in.withShape(kept.stream().mapToLong(Long::longValue).toArray())),
                        List.of(
                                new NamedAttribute(
                                        "dimensions",
                                        DenseArrayAttr.i64(
                                                dims.stream()
                                                        .mapToLong(Long::longValue)
                                                        .toArray()))),
                        List.of(),
                        List.of(Region.of(body)));
        block.addOperation(reduce);
        return reshape(reduce.result(0), node.shape().flattenModes().toArray());
    }

    private static long identity(ReductionOperator op, DataType dt) {
        if (dt == DataType.BOOL) {
            return switch (op) {
                case SUM, MAX -> 0;
                case PROD, MIN -> 1;
            };
        }
        if (dt.isFloatingPoint()) {
            FloatFormat f = ((FloatType) elementType(dt)).format();
            return switch (op) {
                case SUM -> f.encode(0.0);
                case PROD -> f.encode(1.0);
                case MAX -> f.encode(Double.NEGATIVE_INFINITY);
                case MIN -> f.encode(Double.POSITIVE_INFINITY);
            };
        }
        int bits = (int) dt.byteSize() * 8;
        long min = bits == 64 ? Long.MIN_VALUE : -(1L << (bits - 1));
        long max = bits == 64 ? Long.MAX_VALUE : (1L << (bits - 1)) - 1;
        return switch (op) {
            case SUM -> 0;
            case PROD -> 1;
            case MAX -> min;
            case MIN -> max;
        };
    }

    @Override
    public Value visitGatherOp(GatherOp node) {
        // take along one axis: output = input dims before axis ++ indices dims ++ dims after
        Value input = value(node.input());
        Value indices = value(node.indices());
        TensorType in = type(input);
        int r = in.rank();
        int q = type(indices).rank();
        int a = Util.wrapAround(node.axis(), r);
        long[] offsetDims = new long[r - 1];
        for (int i = 0, k = 0; i < r; i++) {
            if (i != a) {
                offsetDims[k++] = i < a ? i : i - 1 + q;
            }
        }
        long[] sliceSizes = in.dims();
        sliceSizes[a] = 1;
        GatherDimensionNumbers dn =
                new GatherDimensionNumbers(
                        offsetDims, new long[] {a}, new long[0], new long[0], new long[] {a}, q);
        return emit(
                "stablehlo.gather",
                List.of(input, indices),
                tensorType(node.shape(), node.dataType()),
                new NamedAttribute("dimension_numbers", dn.toAttribute()),
                new NamedAttribute("slice_sizes", DenseArrayAttr.i64(sliceSizes)));
    }

    // ================================================================== views

    @Override
    public Value visitContiguous(Contiguous node) {
        return value(node.input()); // layouts do not exist in StableHLO
    }

    @Override
    public Value visitViewTransform(ViewTransform node) {
        Value x = value(node.input());
        long[] out = node.shape().flattenModes().toArray();
        return switch (node.operation()) {
            case ViewOperation.Transpose t -> {
                long[] perm = Arrays.stream(t.permutation()).asLongStream().toArray();
                yield emit(
                        "stablehlo.transpose",
                        List.of(x),
                        type(x).withShape(out),
                        new NamedAttribute("permutation", DenseArrayAttr.i64(perm)));
            }
            case ViewOperation.Reshape ignored -> reshape(x, out);
            case ViewOperation.Unsqueeze ignored -> reshape(x, out);
            case ViewOperation.Broadcast ignored -> broadcastTo(x, node.shape());
            case ViewOperation.Expand ignored -> broadcastTo(x, node.shape());
            case ViewOperation.Slice s -> slice(x, s, out);
        };
    }

    private Value slice(Value x, ViewOperation.Slice s, long[] out) {
        TensorType t = type(x);
        int axis = Util.wrapAround(s.axis(), t.rank());
        long n = t.dim(axis);
        long length = out[axis];
        long step = s.step();
        long start = s.start() < 0 ? s.start() + n : s.start();
        if (step < 0) { // reverse the axis, then slice forward
            x =
                    emit(
                            "stablehlo.reverse",
                            List.of(x),
                            t,
                            new NamedAttribute("dimensions", DenseArrayAttr.i64(axis)));
            start = n - 1 - start;
            step = -step;
        }
        long[] starts = new long[t.rank()];
        long[] limits = t.dims();
        long[] strides = new long[t.rank()];
        Arrays.fill(strides, 1);
        starts[axis] = length == 0 ? 0 : start;
        limits[axis] = length == 0 ? 0 : start + (length - 1) * step + 1;
        strides[axis] = step;
        return emit(
                "stablehlo.slice",
                List.of(x),
                t.withShape(out),
                new NamedAttribute("start_indices", DenseArrayAttr.i64(starts)),
                new NamedAttribute("limit_indices", DenseArrayAttr.i64(limits)),
                new NamedAttribute("strides", DenseArrayAttr.i64(strides)));
    }
}

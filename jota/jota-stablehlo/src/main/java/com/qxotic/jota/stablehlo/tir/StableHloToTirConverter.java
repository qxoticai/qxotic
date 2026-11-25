package com.qxotic.jota.stablehlo.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.BinaryOp;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.ir.tir.CastOp;
import com.qxotic.jota.ir.tir.GatherOp;
import com.qxotic.jota.ir.tir.IotaConstant;
import com.qxotic.jota.ir.tir.ReductionOp;
import com.qxotic.jota.ir.tir.ReductionOperator;
import com.qxotic.jota.ir.tir.ScalarConstant;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.ir.tir.TernaryOp;
import com.qxotic.jota.ir.tir.TernaryOperator;
import com.qxotic.jota.ir.tir.UnaryOp;
import com.qxotic.jota.ir.tir.UnaryOperator;
import com.qxotic.jota.ir.tir.ViewOperation;
import com.qxotic.jota.ir.tir.ViewTransform;
import com.qxotic.jota.stablehlo.attr.ComparisonDirection;
import com.qxotic.jota.stablehlo.attr.ComparisonType;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.op.*;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Converts a StableHLO function into a TIR graph ({@code jota-tensor}'s tensor IR).
 *
 * <p>Implemented as an {@link OpVisitor}: each supported op maps to one TIR node or to a small
 * composition of TIR nodes (see {@code jota/TODO-parity-TIR-StableHLO.md}, sections 4 and 6), any
 * other op raises {@link UnsupportedStableHloException}. Calls and composites are inlined; {@code
 * check.*} custom calls (test assertions) are ignored.
 *
 * <p>Semantics follow the StableHLO spec where TIR differs: shifts by amounts outside {@code [0,
 * bits)} give 0 (or the sign for arithmetic shifts), float→bool conversion tests {@code != 0},
 * {@code NE/LE/GE} comparisons are NaN-correct, gather and dynamic slice indices are clamped.
 *
 * <p>Not supported: control flow ({@code while}, {@code if}, {@code case}), {@code scatter}, {@code
 * sort}, convolution and windowed reductions, multi-input reductions, rounding ops, unsigned /
 * complex / fp8 / quantized types, dynamic shapes, collectives and side effects.
 */
public final class StableHloToTirConverter implements OpVisitor<List<TIRNode>> {

    private static final int MAX_INLINE_DEPTH = 64;

    private final Program program;
    private final Map<Value, TIRNode> env = new IdentityHashMap<>();
    private final List<TIRNode> inputs = new ArrayList<>();
    private final List<TirConversion.InputBinding> bindings = new ArrayList<>();
    private int inlineDepth;

    private StableHloToTirConverter(Program program) {
        this.program = program;
    }

    /** Converts function {@code functionName} of {@code program}. */
    public static TirConversion convert(Program program, String functionName) {
        FuncOp func =
                program.function(functionName)
                        .orElseThrow(
                                () -> new IllegalArgumentException("no function @" + functionName));
        StableHloToTirConverter c = new StableHloToTirConverter(program);
        Block body = func.body().block();
        for (int i = 0; i < body.arguments().size(); i++) {
            Value arg = body.argument(i);
            TensorInput in =
                    new TensorInput(
                            c.inputs.size(),
                            TirTypes.dataType(arg.type(), "func.func"),
                            Layout.rowMajor(TirTypes.shape(arg.type(), "func.func")));
            c.inputs.add(in);
            c.bindings.add(new TirConversion.Argument(i));
            c.env.put(arg, in);
        }
        List<TIRNode> outputs = c.run(body);
        List<TensorType> resultTypes = new ArrayList<>();
        for (var t :
                ((com.qxotic.jota.stablehlo.type.FunctionType) func.functionType()).results()) {
            resultTypes.add(TirTypes.tensorType(t, "func.func"));
        }
        return new TirConversion(new TIRGraph(c.inputs, outputs), c.bindings, resultTypes);
    }

    /** Converts the operations of a block, returning the values of its terminator. */
    private List<TIRNode> run(Block block) {
        for (Operation op : block.operations()) {
            if (op.name().equals("func.return") || op.name().equals("stablehlo.return")) {
                return op.operands().stream().map(this::node).toList();
            }
            List<TIRNode> results = op.asOp().accept(this);
            if (results.size() != op.numResults()) {
                throw new IllegalStateException(
                        op.name() + " produced " + results.size() + " TIR results");
            }
            for (int i = 0; i < results.size(); i++) {
                env.put(op.result(i), results.get(i));
            }
        }
        throw new UnsupportedStableHloException("block", "missing terminator");
    }

    private TIRNode node(Value v) {
        TIRNode n = env.get(v);
        if (n == null) {
            throw new IllegalStateException("value used before definition: " + v);
        }
        return n;
    }

    private static List<TIRNode> one(TIRNode n) {
        return List.of(n);
    }

    // ================================================================== defaults

    @Override
    public List<TIRNode> visitDefault(Op op) {
        throw new UnsupportedStableHloException(op.name(), "no TIR translation");
    }

    @Override
    public List<TIRNode> visitGeneric(GenericOp op) {
        if (op.operation().dialect().equals("check") && op.operation().numResults() == 0) {
            return List.of(); // test assertions
        }
        return visitDefault(op);
    }

    @Override
    public List<TIRNode> visitCustomCall(CustomCallOp op) {
        if (op.callTargetName().startsWith("check.") && op.operation().numResults() == 0) {
            return List.of(); // test assertions
        }
        return visitDefault(op);
    }

    @Override
    public List<TIRNode> visitOptimizationBarrier(OptimizationBarrierOp op) {
        return op.operand().stream().map(this::node).toList();
    }

    @Override
    public List<TIRNode> visitCall(CallOp op) {
        return inline(op.name(), op.callee(), op.inputs());
    }

    @Override
    public List<TIRNode> visitComposite(CompositeOp op) {
        return inline(op.name(), op.decomposition(), op.inputs());
    }

    private List<TIRNode> inline(String opName, String callee, List<Value> args) {
        FuncOp f =
                program.function(callee)
                        .orElseThrow(
                                () ->
                                        new UnsupportedStableHloException(
                                                opName, "unknown function @" + callee));
        if (++inlineDepth > MAX_INLINE_DEPTH) {
            throw new UnsupportedStableHloException(opName, "recursion too deep at @" + callee);
        }
        try {
            Block body = f.body().block();
            for (int i = 0; i < args.size(); i++) {
                env.put(body.argument(i), node(args.get(i)));
            }
            return run(body);
        } finally {
            inlineDepth--;
        }
    }

    // ================================================================== constants

    @Override
    public List<TIRNode> visitConstant(ConstantOp op) {
        if (!(op.value() instanceof DenseElementsAttr dense)) {
            throw new UnsupportedStableHloException(op.name(), "non-dense value " + op.value());
        }
        TensorType tt = TirTypes.tensorType(op.result().type(), op.name());
        DataType dt = TirTypes.dataType(tt.elementType(), op.name());
        Shape shape = Shape.flat(tt.dims());
        if (dense.isSplat()) {
            return one(ScalarConstant.broadcast(dense.rawBits(0), dt, shape));
        }
        TensorInput in = new TensorInput(inputs.size(), dt, Layout.rowMajor(shape));
        inputs.add(in);
        bindings.add(new TirConversion.Constant(dense));
        return one(in);
    }

    @Override
    public List<TIRNode> visitIota(IotaOp op) {
        Shape shape = TirTypes.shape(op.result().type(), op.name());
        DataType dt = TirTypes.dataType(op.result().type(), op.name());
        return one(iotaAlong(shape, (int) op.iotaDimension(), dt));
    }

    @Override
    public List<TIRNode> visitGetDimensionSize(GetDimensionSizeOp op) {
        Shape shape = TirTypes.shape(op.operand().type(), op.name());
        return one(ScalarConstant.of(shape.flatAt((int) op.dimension()), DataType.I32));
    }

    // ================================================================== elementwise unary

    private List<TIRNode> unary(Op op, Value x, UnaryOperator u) {
        return one(new UnaryOp(u, node(x)));
    }

    @Override
    public List<TIRNode> visitAbs(AbsOp op) {
        return unary(op, op.operand(), UnaryOperator.ABS);
    }

    @Override
    public List<TIRNode> visitNegate(NegateOp op) {
        return unary(op, op.operand(), UnaryOperator.NEGATE);
    }

    @Override
    public List<TIRNode> visitExponential(ExponentialOp op) {
        return unary(op, op.operand(), UnaryOperator.EXP);
    }

    @Override
    public List<TIRNode> visitLog(LogOp op) {
        return unary(op, op.operand(), UnaryOperator.LOG);
    }

    @Override
    public List<TIRNode> visitSqrt(SqrtOp op) {
        return unary(op, op.operand(), UnaryOperator.SQRT);
    }

    @Override
    public List<TIRNode> visitSine(SineOp op) {
        return unary(op, op.operand(), UnaryOperator.SIN);
    }

    @Override
    public List<TIRNode> visitCosine(CosineOp op) {
        return unary(op, op.operand(), UnaryOperator.COS);
    }

    @Override
    public List<TIRNode> visitTan(TanOp op) {
        return unary(op, op.operand(), UnaryOperator.TAN);
    }

    @Override
    public List<TIRNode> visitTanh(TanhOp op) {
        return unary(op, op.operand(), UnaryOperator.TANH);
    }

    @Override
    public List<TIRNode> visitRsqrt(RsqrtOp op) {
        return one(
                new UnaryOp(
                        UnaryOperator.RECIPROCAL,
                        new UnaryOp(UnaryOperator.SQRT, node(op.operand()))));
    }

    @Override
    public List<TIRNode> visitLogistic(LogisticOp op) {
        TIRNode x = node(op.operand());
        TIRNode e = new UnaryOp(UnaryOperator.EXP, new UnaryOp(UnaryOperator.NEGATE, x));
        return one(
                new UnaryOp(
                        UnaryOperator.RECIPROCAL,
                        bin(BinaryOperator.ADD, constantLike(x, 1.0), e)));
    }

    @Override
    public List<TIRNode> visitExponentialMinusOne(ExponentialMinusOneOp op) {
        TIRNode x = node(op.operand());
        return one(
                bin(
                        BinaryOperator.SUBTRACT,
                        new UnaryOp(UnaryOperator.EXP, x),
                        constantLike(x, 1.0)));
    }

    @Override
    public List<TIRNode> visitLogPlusOne(LogPlusOneOp op) {
        TIRNode x = node(op.operand());
        return one(
                new UnaryOp(UnaryOperator.LOG, bin(BinaryOperator.ADD, x, constantLike(x, 1.0))));
    }

    @Override
    public List<TIRNode> visitNot(NotOp op) {
        TIRNode x = node(op.operand());
        return one(
                new UnaryOp(
                        x.dataType() == DataType.BOOL
                                ? UnaryOperator.LOGICAL_NOT
                                : UnaryOperator.BITWISE_NOT,
                        x));
    }

    @Override
    public List<TIRNode> visitSign(SignOp op) {
        TIRNode x = node(op.operand());
        TIRNode zero = constantLike(x, 0);
        TIRNode positive = bin(BinaryOperator.LESS_THAN, zero, x);
        TIRNode negative = bin(BinaryOperator.LESS_THAN, x, zero);
        // NaN and ±0 fall through to x itself
        return one(where(positive, constantLike(x, 1), where(negative, constantLike(x, -1), x)));
    }

    @Override
    public List<TIRNode> visitIsFinite(IsFiniteOp op) {
        TIRNode x = node(op.operand());
        return one(
                bin(
                        BinaryOperator.LESS_THAN,
                        new UnaryOp(UnaryOperator.ABS, x),
                        constantLike(x, Double.POSITIVE_INFINITY)));
    }

    @Override
    public List<TIRNode> visitConvert(ConvertOp op) {
        TIRNode x = node(op.operand());
        DataType target = TirTypes.dataType(op.result().type(), op.name());
        if (x.dataType() == target) {
            return one(x);
        }
        if (target == DataType.BOOL) { // spec: non-zero (and NaN) → true
            return one(not(bin(BinaryOperator.EQUAL, x, constantLike(x, 0))));
        }
        return one(new CastOp(x, target));
    }

    // ================================================================== elementwise binary

    private List<TIRNode> binary(Op op, Value lhs, Value rhs, BinaryOperator b) {
        return one(bin(b, node(lhs), node(rhs)));
    }

    private List<TIRNode> boolOr(
            Op op, Value lhs, Value rhs, BinaryOperator forBool, BinaryOperator otherwise) {
        TIRNode l = node(lhs);
        return one(bin(l.dataType() == DataType.BOOL ? forBool : otherwise, l, node(rhs)));
    }

    @Override
    public List<TIRNode> visitAdd(AddOp op) {
        return boolOr(op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_OR, BinaryOperator.ADD);
    }

    @Override
    public List<TIRNode> visitSubtract(SubtractOp op) {
        return binary(op, op.lhs(), op.rhs(), BinaryOperator.SUBTRACT);
    }

    @Override
    public List<TIRNode> visitMultiply(MultiplyOp op) {
        return boolOr(op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_AND, BinaryOperator.MULTIPLY);
    }

    @Override
    public List<TIRNode> visitDivide(DivideOp op) {
        return binary(op, op.lhs(), op.rhs(), BinaryOperator.DIVIDE);
    }

    @Override
    public List<TIRNode> visitMaximum(MaximumOp op) {
        return boolOr(op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_OR, BinaryOperator.MAX);
    }

    @Override
    public List<TIRNode> visitMinimum(MinimumOp op) {
        return boolOr(op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_AND, BinaryOperator.MIN);
    }

    @Override
    public List<TIRNode> visitPower(PowerOp op) {
        TIRNode l = node(op.lhs());
        if (!l.dataType().isFloatingPoint()) {
            throw new UnsupportedStableHloException(op.name(), "integer power");
        }
        return one(bin(BinaryOperator.POW, l, node(op.rhs())));
    }

    @Override
    public List<TIRNode> visitRemainder(RemainderOp op) {
        TIRNode a = node(op.lhs());
        TIRNode b = node(op.rhs());
        if (a.dataType().isFloatingPoint()) {
            throw new UnsupportedStableHloException(op.name(), "floating-point remainder");
        }
        TIRNode q = bin(BinaryOperator.DIVIDE, a, b);
        return one(bin(BinaryOperator.SUBTRACT, a, bin(BinaryOperator.MULTIPLY, q, b)));
    }

    @Override
    public List<TIRNode> visitAnd(AndOp op) {
        return boolOr(
                op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_AND, BinaryOperator.BITWISE_AND);
    }

    @Override
    public List<TIRNode> visitOr(OrOp op) {
        return boolOr(op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_OR, BinaryOperator.BITWISE_OR);
    }

    @Override
    public List<TIRNode> visitXor(XorOp op) {
        return boolOr(
                op, op.lhs(), op.rhs(), BinaryOperator.LOGICAL_XOR, BinaryOperator.BITWISE_XOR);
    }

    @Override
    public List<TIRNode> visitShiftLeft(ShiftLeftOp op) {
        return shift(op.lhs(), op.rhs(), BinaryOperator.SHIFT_LEFT, false);
    }

    @Override
    public List<TIRNode> visitShiftRightLogical(ShiftRightLogicalOp op) {
        return shift(op.lhs(), op.rhs(), BinaryOperator.SHIFT_RIGHT_UNSIGNED, false);
    }

    @Override
    public List<TIRNode> visitShiftRightArithmetic(ShiftRightArithmeticOp op) {
        return shift(op.lhs(), op.rhs(), BinaryOperator.SHIFT_RIGHT, true);
    }

    /** StableHLO shifts: amounts outside {@code [0, bits)} give 0, or the sign for {@code >>}. */
    private List<TIRNode> shift(Value lhs, Value rhs, BinaryOperator b, boolean arithmetic) {
        TIRNode x = node(lhs);
        TIRNode s = node(rhs);
        int bits = TirTypes.bitWidth(x.dataType());
        TIRNode inRange =
                and(
                        not(bin(BinaryOperator.LESS_THAN, s, constantLike(s, 0))),
                        bin(BinaryOperator.LESS_THAN, s, constantLike(s, bits)));
        TIRNode outOfRange =
                arithmetic
                        ? where(
                                bin(BinaryOperator.LESS_THAN, x, constantLike(x, 0)),
                                constantLike(x, -1),
                                constantLike(x, 0))
                        : constantLike(x, 0);
        return one(where(inRange, bin(b, x, s), outOfRange));
    }

    @Override
    public List<TIRNode> visitCompare(CompareOp op) {
        if (op.compareType().orElse(ComparisonType.NOTYPE) == ComparisonType.TOTALORDER) {
            throw new UnsupportedStableHloException(op.name(), "TOTALORDER comparison");
        }
        TIRNode a = node(op.lhs());
        TIRNode b = node(op.rhs());
        ComparisonDirection d = op.comparisonDirection();
        TIRNode r =
                switch (d) {
                    case EQ -> bin(BinaryOperator.EQUAL, a, b);
                    case NE -> not(bin(BinaryOperator.EQUAL, a, b));
                    case LT -> bin(BinaryOperator.LESS_THAN, a, b);
                    case GT -> bin(BinaryOperator.LESS_THAN, b, a);
                    case LE ->
                            or(
                                    bin(BinaryOperator.LESS_THAN, a, b),
                                    bin(BinaryOperator.EQUAL, a, b));
                    case GE ->
                            or(
                                    bin(BinaryOperator.LESS_THAN, b, a),
                                    bin(BinaryOperator.EQUAL, a, b));
                };
        return one(r);
    }

    @Override
    public List<TIRNode> visitSelect(SelectOp op) {
        return one(where(node(op.pred()), node(op.onTrue()), node(op.onFalse())));
    }

    @Override
    public List<TIRNode> visitClamp(ClampOp op) {
        TIRNode x = node(op.operand());
        return one(
                bin(
                        BinaryOperator.MIN,
                        bin(BinaryOperator.MAX, x, node(op.min())),
                        node(op.max())));
    }

    // ================================================================== shape ops

    @Override
    public List<TIRNode> visitReshape(ReshapeOp op) {
        return one(reshape(node(op.operand()), TirTypes.shape(op.result().type(), op.name())));
    }

    @Override
    public List<TIRNode> visitTranspose(TransposeOp op) {
        return one(transpose(node(op.operand()), toInts(op.permutation())));
    }

    @Override
    public List<TIRNode> visitBroadcast(BroadcastOp op) {
        return one(broadcastTo(node(op.operand()), TirTypes.shape(op.result().type(), op.name())));
    }

    @Override
    public List<TIRNode> visitBroadcastInDim(BroadcastInDimOp op) {
        TIRNode x = node(op.operand());
        Shape out = TirTypes.shape(op.result().type(), op.name());
        long[] dims = op.broadcastDimensions();
        int rank = x.shape().flatRank();
        // order operand axes by their target dimension
        Integer[] order = new Integer[rank];
        for (int i = 0; i < rank; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (p, q) -> Long.compare(dims[p], dims[q]));
        int[] perm = Arrays.stream(order).mapToInt(Integer::intValue).toArray();
        TIRNode t = transpose(x, perm);
        long[] withOnes = new long[out.flatRank()];
        Arrays.fill(withOnes, 1);
        for (int k = 0; k < rank; k++) {
            withOnes[(int) dims[perm[k]]] = x.shape().flatAt(perm[k]);
        }
        return one(broadcastTo(reshape(t, Shape.flat(withOnes)), out));
    }

    @Override
    public List<TIRNode> visitSlice(SliceOp op) {
        TIRNode x = node(op.operand());
        long[] start = op.startIndices();
        long[] limit = op.limitIndices();
        long[] strides = op.strides();
        for (int axis = 0; axis < start.length; axis++) {
            long dim = x.shape().flatAt(axis);
            if (start[axis] == 0 && limit[axis] == dim && strides[axis] == 1) {
                continue;
            }
            x =
                    new ViewTransform(
                            x,
                            new ViewOperation.Slice(axis, start[axis], limit[axis], strides[axis]));
        }
        return one(x);
    }

    @Override
    public List<TIRNode> visitReverse(ReverseOp op) {
        TIRNode x = node(op.operand());
        for (long axis : op.dimensions()) {
            long n = x.shape().flatAt((int) axis);
            TIRNode reversed =
                    bin(BinaryOperator.SUBTRACT, ScalarConstant.of(n - 1, DataType.I64), iota1d(n));
            x = gather(x, reversed, (int) axis);
        }
        return one(x);
    }

    @Override
    public List<TIRNode> visitConcatenate(ConcatenateOp op) {
        int axis = (int) op.dimension();
        Shape out = TirTypes.shape(op.result().type(), op.name());
        long m = out.flatAt(axis);
        TIRNode coord = iota1d(m);
        TIRNode result = null;
        long offset = 0;
        for (Value input : op.inputs()) {
            TIRNode x = node(input);
            long n = x.shape().flatAt(axis);
            if (n == 0) {
                continue;
            }
            TIRNode local =
                    clamp(
                            bin(
                                    BinaryOperator.SUBTRACT,
                                    coord,
                                    ScalarConstant.of(offset, DataType.I64)),
                            0,
                            n - 1);
            TIRNode part = gather(x, local, axis);
            if (result == null) {
                result = part;
            } else {
                TIRNode inPart =
                        not(
                                bin(
                                        BinaryOperator.LESS_THAN,
                                        coord,
                                        ScalarConstant.of(offset, DataType.I64)));
                result = where(alongAxis(inPart, out, axis), part, result);
            }
            offset += n;
        }
        if (result == null) {
            throw new UnsupportedStableHloException(op.name(), "all inputs are empty");
        }
        return one(result);
    }

    @Override
    public List<TIRNode> visitPad(PadOp op) {
        TIRNode x = node(op.operand());
        TIRNode padValue = node(op.paddingValue());
        long[] low = op.edgePaddingLow();
        long[] high = op.edgePaddingHigh();
        long[] interior = op.interiorPadding();
        for (int axis = 0; axis < low.length; axis++) {
            if (low[axis] == 0 && high[axis] == 0 && interior[axis] == 0) {
                continue;
            }
            long n = x.shape().flatAt(axis);
            long m = low[axis] + high[axis] + (n == 0 ? 0 : n + (n - 1) * interior[axis]);
            long[] outDims = x.shape().toArray();
            outDims[axis] = m;
            Shape outShape = Shape.flat(outDims);
            if (n == 0) {
                x = broadcastTo(padValue, outShape);
                continue;
            }
            long step = interior[axis] + 1;
            TIRNode coord =
                    bin(
                            BinaryOperator.SUBTRACT,
                            iota1d(m),
                            ScalarConstant.of(low[axis], DataType.I64));
            TIRNode stepC = ScalarConstant.of(step, DataType.I64);
            TIRNode q = bin(BinaryOperator.DIVIDE, coord, stepC);
            TIRNode rem =
                    bin(BinaryOperator.SUBTRACT, coord, bin(BinaryOperator.MULTIPLY, q, stepC));
            TIRNode valid =
                    and(
                            and(
                                    not(
                                            bin(
                                                    BinaryOperator.LESS_THAN,
                                                    coord,
                                                    ScalarConstant.of(0, DataType.I64))),
                                    bin(
                                            BinaryOperator.EQUAL,
                                            rem,
                                            ScalarConstant.of(0, DataType.I64))),
                            bin(BinaryOperator.LESS_THAN, q, ScalarConstant.of(n, DataType.I64)));
            TIRNode gathered = gather(x, clamp(q, 0, n - 1), axis);
            x = where(alongAxis(valid, outShape, axis), gathered, broadcastTo(padValue, outShape));
        }
        return one(x);
    }

    // ================================================================== gather & slicing

    @Override
    public List<TIRNode> visitGather(com.qxotic.jota.stablehlo.op.GatherOp op) {
        GatherDimensionNumbers dn = op.dimensionNumbers();
        TIRNode operand = node(op.operand());
        TIRNode startIndices = node(op.startIndices());
        Shape out = TirTypes.shape(op.result().type(), op.name());
        int r = out.flatRank();
        long[] offsetDims = dn.offsetDims();
        List<Integer> batchDims = new ArrayList<>();
        for (int a = 0; a < r; a++) {
            if (!contains(offsetDims, a)) {
                batchDims.add(a);
            }
        }
        int sRank = startIndices.shape().flatRank();
        long ivd = dn.indexVectorDim();
        Shape operandShape = operand.shape();
        int n = operandShape.flatRank();
        long[] sliceSizes = op.sliceSizes();
        List<Integer> offsetOperandDims = new ArrayList<>();
        for (int d = 0; d < n; d++) {
            if (!contains(dn.collapsedSliceDims(), d) && !contains(dn.operandBatchingDims(), d)) {
                offsetOperandDims.add(d);
            }
        }
        TIRNode linear = null;
        long stride = 1;
        for (int d = n - 1; d >= 0; d--) {
            TIRNode idx = null;
            int k = indexOf(dn.startIndexMap(), d);
            if (k >= 0) {
                TIRNode component = startComponent(startIndices, sRank, ivd, k);
                component = clamp(component, 0, operandShape.flatAt(d) - sliceSizes[d]);
                idx = toResultShape(component, out, batchDims);
            }
            int bi = indexOf(dn.operandBatchingDims(), d);
            if (bi >= 0) {
                long dStart = dn.startIndicesBatchingDims()[bi];
                int pos = (int) (ivd < sRank && dStart > ivd ? dStart - 1 : dStart);
                idx = add(idx, iotaAlong(out, batchDims.get(pos), DataType.I64));
            }
            int oi = offsetOperandDims.indexOf(d);
            if (oi >= 0) {
                idx = add(idx, iotaAlong(out, (int) offsetDims[oi], DataType.I64));
            }
            if (idx != null) {
                TIRNode term =
                        stride == 1
                                ? idx
                                : bin(
                                        BinaryOperator.MULTIPLY,
                                        idx,
                                        ScalarConstant.of(stride, DataType.I64));
                linear = add(linear, term);
            }
            stride *= operandShape.flatAt(d);
        }
        if (linear == null) {
            linear = ScalarConstant.broadcast(0, DataType.I64, out);
        }
        linear = broadcastTo(linear, out);
        return one(gather(reshape(operand, Shape.flat(operandShape.size())), linear, 0));
    }

    /** Component {@code k} of the index vectors, as an I64 tensor over the batch dimensions. */
    private TIRNode startComponent(TIRNode startIndices, int sRank, long ivd, int k) {
        TIRNode c = startIndices;
        if (ivd < sRank) {
            c = new ViewTransform(c, new ViewOperation.Slice((int) ivd, k, k + 1, 1));
            long[] dims = c.shape().toArray();
            long[] without = new long[dims.length - 1];
            for (int i = 0, j = 0; i < dims.length; i++) {
                if (i != ivd) {
                    without[j++] = dims[i];
                }
            }
            c = reshape(c, Shape.flat(without));
        }
        return toI64(c);
    }

    /** A tensor over the batch dimensions of {@code out}, broadcast to {@code out}. */
    private TIRNode toResultShape(TIRNode batchValues, Shape out, List<Integer> batchDims) {
        long[] dims = new long[out.flatRank()];
        Arrays.fill(dims, 1);
        for (int b : batchDims) {
            dims[b] = out.flatAt(b);
        }
        return broadcastTo(reshape(batchValues, Shape.flat(dims)), out);
    }

    @Override
    public List<TIRNode> visitDynamicSlice(DynamicSliceOp op) {
        TIRNode operand = node(op.operand());
        Shape in = operand.shape();
        Shape out = TirTypes.shape(op.result().type(), op.name());
        TIRNode linear = null;
        long stride = 1;
        for (int d = in.flatRank() - 1; d >= 0; d--) {
            TIRNode start =
                    clamp(toI64(node(op.startIndices().get(d))), 0, in.flatAt(d) - out.flatAt(d));
            TIRNode idx = add(start, iotaAlong(out, d, DataType.I64));
            linear = add(linear, scale(idx, stride));
            stride *= in.flatAt(d);
        }
        return one(gather(reshape(operand, Shape.flat(in.size())), broadcastTo(linear, out), 0));
    }

    @Override
    public List<TIRNode> visitDynamicUpdateSlice(DynamicUpdateSliceOp op) {
        TIRNode operand = node(op.operand());
        TIRNode update = node(op.update());
        Shape in = operand.shape();
        Shape up = update.shape();
        if (up.size() == 0) { // empty update: nothing changes
            return one(operand);
        }
        TIRNode inside = null;
        TIRNode linear = null;
        long stride = 1;
        for (int d = in.flatRank() - 1; d >= 0; d--) {
            TIRNode start =
                    clamp(toI64(node(op.startIndices().get(d))), 0, in.flatAt(d) - up.flatAt(d));
            TIRNode rel = bin(BinaryOperator.SUBTRACT, iotaAlong(in, d, DataType.I64), start);
            TIRNode ok =
                    and(
                            not(
                                    bin(
                                            BinaryOperator.LESS_THAN,
                                            rel,
                                            ScalarConstant.of(0, DataType.I64))),
                            bin(
                                    BinaryOperator.LESS_THAN,
                                    rel,
                                    ScalarConstant.of(up.flatAt(d), DataType.I64)));
            inside = inside == null ? ok : and(inside, ok);
            linear = add(linear, scale(clamp(rel, 0, Math.max(0, up.flatAt(d) - 1)), stride));
            stride *= up.flatAt(d);
        }
        if (inside == null) { // rank 0: the update replaces the operand
            return one(update);
        }
        TIRNode gathered =
                gather(reshape(update, Shape.flat(up.size())), broadcastTo(linear, in), 0);
        return one(where(inside, gathered, operand));
    }

    // ================================================================== contractions & reductions

    @Override
    public List<TIRNode> visitDot(DotOp op) {
        TIRNode l = node(op.lhs());
        TIRNode r = node(op.rhs());
        long[] lc = {l.shape().flatRank() - 1};
        long[] rc = {0};
        return one(
                dotGeneral(
                        op.name(),
                        l,
                        r,
                        new DotDimensionNumbers(new long[0], new long[0], lc, rc),
                        op.result().type()));
    }

    @Override
    public List<TIRNode> visitDotGeneral(DotGeneralOp op) {
        if (op.operation().numOperands() != 2) {
            throw new UnsupportedStableHloException(op.name(), "extra operands");
        }
        return one(
                dotGeneral(
                        op.name(),
                        node(op.lhs()),
                        node(op.rhs()),
                        op.dotDimensionNumbers(),
                        op.result().type()));
    }

    /**
     * {@code [B, LF, C] x [B, RF, C]}: transpose both sides to batch / free / contracting order,
     * broadcast to {@code [B, LF, RF, C]}, multiply in the result type, sum over {@code C}.
     */
    private TIRNode dotGeneral(
            String opName,
            TIRNode lhs,
            TIRNode rhs,
            DotDimensionNumbers dn,
            com.qxotic.jota.stablehlo.type.Type resultType) {
        DataType acc = TirTypes.dataType(resultType, opName);
        int[] lb = toInts(dn.lhsBatchingDimensions());
        int[] rb = toInts(dn.rhsBatchingDimensions());
        int[] lc = toInts(dn.lhsContractingDimensions());
        int[] rc = toInts(dn.rhsContractingDimensions());
        int[] lf = free(lhs.shape().flatRank(), lb, lc);
        int[] rf = free(rhs.shape().flatRank(), rb, rc);
        TIRNode lt = transpose(lhs, concat(lb, lf, lc));
        TIRNode rt = transpose(rhs, concat(rb, rf, rc));
        long[] batch = dims(lhs.shape(), lb);
        long[] lfree = dims(lhs.shape(), lf);
        long[] rfree = dims(rhs.shape(), rf);
        long[] contract = dims(lhs.shape(), lc);
        long[] full = concat(batch, lfree, rfree, contract);
        TIRNode lr = reshape(lt, Shape.flat(concat(batch, lfree, ones(rfree.length), contract)));
        TIRNode rr = reshape(rt, Shape.flat(concat(batch, ones(lfree.length), rfree, contract)));
        TIRNode product =
                bin(
                        BinaryOperator.MULTIPLY,
                        cast(broadcastTo(lr, Shape.flat(full)), acc),
                        cast(broadcastTo(rr, Shape.flat(full)), acc));
        if (contract.length == 0) {
            return product;
        }
        int[] axes = new int[contract.length];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = full.length - contract.length + i;
        }
        return new ReductionOp(ReductionOperator.SUM, product, axes, false, acc);
    }

    @Override
    public List<TIRNode> visitReduce(ReduceOp op) {
        if (op.inputs().size() != 1) {
            throw new UnsupportedStableHloException(op.name(), "variadic reduce");
        }
        Block body = op.body().block();
        List<Operation> ops = body.operations();
        if (ops.size() != 2 || !ops.get(1).name().equals("stablehlo.return")) {
            throw new UnsupportedStableHloException(op.name(), "reduction body is not a single op");
        }
        Operation combiner = ops.get(0);
        boolean usesArgs =
                combiner.numOperands() == 2
                        && combiner.operands().containsAll(body.arguments())
                        && ops.get(1).operand(0) == combiner.result(0);
        if (!usesArgs) {
            throw new UnsupportedStableHloException(op.name(), "unrecognized reduction body");
        }
        TIRNode x = node(op.inputs().getFirst());
        TIRNode init = node(op.initValues().getFirst());
        DataType acc = TirTypes.dataType(op.initValues().getFirst().type(), op.name());
        boolean bool = acc == DataType.BOOL;
        ReductionOperator reduction;
        BinaryOperator combine;
        switch (combiner.mnemonic()) {
            case "add" -> {
                reduction = bool ? ReductionOperator.MAX : ReductionOperator.SUM;
                combine = bool ? BinaryOperator.LOGICAL_OR : BinaryOperator.ADD;
            }
            case "multiply" -> {
                reduction = bool ? ReductionOperator.MIN : ReductionOperator.PROD;
                combine = bool ? BinaryOperator.LOGICAL_AND : BinaryOperator.MULTIPLY;
            }
            case "maximum", "or" -> {
                reduction = ReductionOperator.MAX;
                combine = bool ? BinaryOperator.LOGICAL_OR : BinaryOperator.MAX;
            }
            case "minimum", "and" -> {
                reduction = ReductionOperator.MIN;
                combine = bool ? BinaryOperator.LOGICAL_AND : BinaryOperator.MIN;
            }
            default ->
                    throw new UnsupportedStableHloException(
                            op.name(), "reduction with " + combiner.name());
        }
        if (combiner.mnemonic().equals("and") || combiner.mnemonic().equals("or")) {
            if (!bool) {
                throw new UnsupportedStableHloException(op.name(), "bitwise reduction");
            }
        }
        int[] axes = toInts(op.dimensions());
        TIRNode reduced =
                axes.length == 0 ? x : new ReductionOp(reduction, cast(x, acc), axes, false, acc);
        return one(bin(combine, reduced, init));
    }

    // ================================================================== helpers

    /**
     * Binary node with explicit broadcasting: TIR's {@code BinaryOp} computes the numpy broadcast
     * shape, but evaluators expect operands of that shape, so they are broadcast as views.
     */
    private static TIRNode bin(BinaryOperator op, TIRNode a, TIRNode b) {
        Shape s = broadcastShape(a.shape(), b.shape());
        return new BinaryOp(op, broadcastTo(a, s), broadcastTo(b, s));
    }

    private static TIRNode where(TIRNode cond, TIRNode t, TIRNode f) {
        Shape s = broadcastShape(cond.shape(), broadcastShape(t.shape(), f.shape()));
        return new TernaryOp(
                TernaryOperator.WHERE, broadcastTo(cond, s), broadcastTo(t, s), broadcastTo(f, s));
    }

    private static Shape broadcastShape(Shape a, Shape b) {
        int ra = a.flatRank();
        int rb = b.flatRank();
        int r = Math.max(ra, rb);
        long[] out = new long[r];
        for (int i = 0; i < r; i++) {
            long da = i < ra ? a.flatAt(ra - 1 - i) : 1;
            long db = i < rb ? b.flatAt(rb - 1 - i) : 1;
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("shapes not broadcastable: " + a + ", " + b);
            }
            out[r - 1 - i] = da == 1 ? db : da;
        }
        return Shape.flat(out);
    }

    private static TIRNode not(TIRNode x) {
        return new UnaryOp(UnaryOperator.LOGICAL_NOT, x);
    }

    private static TIRNode and(TIRNode a, TIRNode b) {
        return bin(BinaryOperator.LOGICAL_AND, a, b);
    }

    private static TIRNode or(TIRNode a, TIRNode b) {
        return bin(BinaryOperator.LOGICAL_OR, a, b);
    }

    private static TIRNode add(TIRNode a, TIRNode b) {
        if (a == null) {
            return b;
        }
        return b == null ? a : bin(BinaryOperator.ADD, a, b);
    }

    private static TIRNode scale(TIRNode x, long factor) {
        return factor == 1
                ? x
                : bin(BinaryOperator.MULTIPLY, x, ScalarConstant.of(factor, DataType.I64));
    }

    private static TIRNode cast(TIRNode x, DataType dt) {
        return x.dataType() == dt ? x : new CastOp(x, dt);
    }

    private static TIRNode toI64(TIRNode x) {
        return cast(x, DataType.I64);
    }

    private static TIRNode clamp(TIRNode x, long lo, long hi) {
        return bin(
                BinaryOperator.MIN,
                bin(BinaryOperator.MAX, x, ScalarConstant.of(lo, x.dataType())),
                ScalarConstant.of(hi, x.dataType()));
    }

    /** A rank-0 constant of {@code like}'s data type. */
    private static TIRNode constantLike(TIRNode like, double value) {
        DataType dt = like.dataType();
        long bits;
        if (dt == DataType.FP32) {
            bits = Float.floatToRawIntBits((float) value) & 0xFFFFFFFFL;
        } else if (dt == DataType.FP64) {
            bits = Double.doubleToRawLongBits(value);
        } else if (dt == DataType.FP16) {
            bits = Float.floatToFloat16((float) value) & 0xFFFFL;
        } else if (dt == DataType.BF16) {
            bits = com.qxotic.jota.stablehlo.type.FloatFormat.BF16.encode(value);
        } else if (dt == DataType.BOOL) {
            bits = value != 0 ? 1 : 0;
        } else {
            bits = (long) value;
        }
        return ScalarConstant.of(bits, dt);
    }

    private static TIRNode reshape(TIRNode x, Shape shape) {
        if (x.shape().flattenModes().equals(shape)) {
            return x;
        }
        if (x.shape().flatRank() == 0) {
            return broadcastTo(x, shape);
        }
        return new ViewTransform(x, new ViewOperation.Reshape(shape));
    }

    private static TIRNode broadcastTo(TIRNode x, Shape shape) {
        if (x.shape().flattenModes().equals(shape)) {
            return x;
        }
        return new ViewTransform(x, new ViewOperation.Broadcast(shape));
    }

    private static TIRNode transpose(TIRNode x, int[] perm) {
        boolean identity = true;
        for (int i = 0; i < perm.length; i++) {
            identity &= perm[i] == i;
        }
        return identity || perm.length == 0
                ? x
                : new ViewTransform(x, new ViewOperation.Transpose(perm));
    }

    /** {@code [0, n)} as I64. */
    private static TIRNode iota1d(long n) {
        return new IotaConstant(n, DataType.I64, Shape.flat(n));
    }

    /** Index along {@code axis}, broadcast to {@code shape}. */
    private static TIRNode iotaAlong(Shape shape, int axis, DataType dt) {
        long n = shape.flatAt(axis);
        TIRNode iota = new IotaConstant(n, dt, Shape.flat(n));
        return broadcastTo(alongAxisShape(iota, shape.flatRank(), axis), shape);
    }

    private static TIRNode alongAxisShape(TIRNode vector, int rank, int axis) {
        long[] dims = new long[rank];
        Arrays.fill(dims, 1);
        dims[axis] = vector.shape().flatAt(0);
        return reshape(vector, Shape.flat(dims));
    }

    /** A 1-D tensor laid along {@code axis} of {@code shape} (broadcast). */
    private static TIRNode alongAxis(TIRNode vector, Shape shape, int axis) {
        return broadcastTo(alongAxisShape(vector, shape.flatRank(), axis), shape);
    }

    private static TIRNode gather(TIRNode x, TIRNode indices, int axis) {
        Shape out = GatherOp.computeOutputShape(x.shape(), indices.shape(), axis);
        return new GatherOp(x, indices, axis, out);
    }

    private static boolean contains(long[] values, long v) {
        return indexOf(values, v) >= 0;
    }

    private static int indexOf(long[] values, long v) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] == v) {
                return i;
            }
        }
        return -1;
    }

    private static int[] toInts(long[] values) {
        return Arrays.stream(values).mapToInt(Math::toIntExact).toArray();
    }

    private static int[] free(int rank, int[] batch, int[] contracting) {
        List<Integer> out = new ArrayList<>();
        for (int d = 0; d < rank; d++) {
            final int dd = d;
            if (Arrays.stream(batch).noneMatch(b -> b == dd)
                    && Arrays.stream(contracting).noneMatch(c -> c == dd)) {
                out.add(d);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] concat(int[]... parts) {
        return Arrays.stream(parts).flatMapToInt(Arrays::stream).toArray();
    }

    private static long[] concat(long[]... parts) {
        return Arrays.stream(parts).flatMapToLong(Arrays::stream).toArray();
    }

    private static long[] dims(Shape shape, int[] axes) {
        long[] out = new long[axes.length];
        for (int i = 0; i < axes.length; i++) {
            out[i] = shape.flatAt(axes[i]);
        }
        return out;
    }

    private static long[] ones(int n) {
        long[] out = new long[n];
        Arrays.fill(out, 1);
        return out;
    }
}

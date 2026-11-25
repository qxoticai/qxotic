package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.op.FuncOp;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Interprets StableHLO programs with the semantics of {@code docs/spec.md} (the reference
 * interpreter of {@code stablehlo/reference} settles what the spec leaves open), on statically
 * shaped tensors.
 *
 * <p>Values are {@link Tensor}s, {@link #TOKEN} for tokens and {@code List<Object>} for tuples.
 * Region-based ops ({@code reduce}, {@code scatter}, {@code sort}...) run their region on rank-0
 * tensors, so any body works. The {@code check.*} operations of StableHLO's interpreter tests (and
 * the {@code check.expect_*} custom calls of {@code testdata}) are executed as assertions: a failed
 * check throws {@link CheckFailure}.
 *
 * <p>Not supported (an {@link InterpreterException}): dynamic shapes, quantized types, collectives,
 * infeed / outfeed / send / recv, fft, triangular_solve, cholesky, rng, the CHLO operations, and
 * other custom calls.
 */
public final class Interpreter {

    /** The value of a {@code !stablehlo.token}. */
    public static final Object TOKEN =
            new Object() {
                @Override
                public String toString() {
                    return "token";
                }
            };

    /** A failed {@code check.*} assertion. */
    public static final class CheckFailure extends RuntimeException {
        CheckFailure(String message) {
            super(message);
        }
    }

    /**
     * Results the spec leaves implementation-defined: integer division and remainder by zero, and
     * {@code MIN / -1}.
     */
    public enum Semantics {
        /** XLA's ({@code x / 0 = -1}, {@code x % 0 = x}, {@code MIN / -1 = MIN}): testdata. */
        XLA,
        /** StableHLO's reference interpreter's (all 0): its {@code tests/interpret} files. */
        REFERENCE
    }

    private final Program program;
    private final Semantics semantics;

    private Interpreter(Program program, Semantics semantics) {
        this.program = program;
        this.semantics = semantics;
    }

    /** Runs function {@code name} of {@code program} on {@code args}; returns its results. */
    public static List<Object> run(Program program, String name, List<Object> args) {
        return run(program, name, args, Semantics.XLA);
    }

    /** As {@link #run(Program, String, List)} with the given implementation-defined results. */
    public static List<Object> run(
            Program program, String name, List<Object> args, Semantics semantics) {
        return new Interpreter(program, semantics).call(name, args);
    }

    /** Runs function {@code name} without arguments. */
    public static List<Object> run(Program program, String name) {
        return run(program, name, List.of());
    }

    private List<Object> call(String name, List<Object> args) {
        FuncOp f =
                program.function(name)
                        .orElseThrow(() -> new InterpreterException("no function @" + name));
        if (f.body().isEmpty()) {
            throw new InterpreterException("function @" + name + " has no body");
        }
        return region(f.body(), args);
    }

    // ================================================================== regions

    private List<Object> region(Region region, List<Object> args) {
        if (region.blocks().size() != 1) {
            throw new InterpreterException("regions with several blocks are not supported");
        }
        Block block = region.block();
        Map<Value, Object> env = new IdentityHashMap<>();
        if (block.arguments().size() != args.size()) {
            throw new InterpreterException(
                    "region takes " + block.arguments().size() + " arguments, got " + args.size());
        }
        for (int i = 0; i < args.size(); i++) {
            env.put(block.argument(i), args.get(i));
        }
        for (Operation op : block.operations()) {
            String name = op.name();
            if (name.equals("func.return") || name.equals("stablehlo.return")) {
                return op.operands().stream().map(v -> lookup(env, v)).toList();
            }
            List<Object> in = op.operands().stream().map(v -> lookup(env, v)).toList();
            List<Object> out = execute(op, in);
            if (out.size() != op.numResults()) {
                throw new InterpreterException(
                        name + " produced " + out.size() + " values for " + op.numResults());
            }
            for (int i = 0; i < out.size(); i++) {
                env.put(op.result(i), out.get(i));
            }
        }
        throw new InterpreterException("region without a return");
    }

    private static Object lookup(Map<Value, Object> env, Value v) {
        Object o = env.get(v);
        if (o == null) {
            throw new InterpreterException("value used before it is computed");
        }
        return o;
    }

    /** Runs a region on scalars (rank-0 tensors); returns its tensor results. */
    private List<Tensor> scalarRegion(Region r, List<Tensor> args) {
        List<Object> res = region(r, new ArrayList<>(args));
        return res.stream().map(o -> (Tensor) o).toList();
    }

    // ================================================================== dispatch

    private static final Set<String> UNARY =
            Set.of(
                    "abs",
                    "negate",
                    "exponential",
                    "exponential_minus_one",
                    "log",
                    "log_plus_one",
                    "logistic",
                    "sqrt",
                    "rsqrt",
                    "cbrt",
                    "sine",
                    "cosine",
                    "tan",
                    "tanh",
                    "floor",
                    "ceil",
                    "round_nearest_even",
                    "round_nearest_afz",
                    "sign",
                    "not",
                    "popcnt",
                    "count_leading_zeros",
                    "is_finite",
                    "real",
                    "imag");

    private static final Set<String> BINARY =
            Set.of(
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "remainder",
                    "power",
                    "atan2",
                    "maximum",
                    "minimum",
                    "and",
                    "or",
                    "xor",
                    "shift_left",
                    "shift_right_logical",
                    "shift_right_arithmetic",
                    "complex");

    private List<Object> execute(Operation op, List<Object> in) {
        String name = op.name();
        if (name.equals("func.call")) {
            return call(((SymbolRefAttr) attr(op, "callee")).root(), in);
        }
        if (name.startsWith("check.")) {
            Checks.check(name, op, in);
            return List.of();
        }
        if (op.dialect().equals("chlo")) {
            return Chlo.execute(op, in);
        }
        if (!op.dialect().equals("stablehlo")) {
            throw new InterpreterException(name + " is not supported");
        }
        String m = op.mnemonic();
        if (UNARY.contains(m)) {
            if ((m.equals("real") || m.equals("imag")) && !t(in, 0).isComplex()) {
                Tensor x = t(in, 0);
                return List.of(
                        m.equals("real")
                                ? x.withType(resultType(op, 0))
                                : Tensor.ofDoubles(resultType(op, 0), new double[x.size()]));
            }
            return List.of(Elementwise.unary(m, t(in, 0), resultType(op, 0)));
        }
        if (BINARY.contains(m)) {
            return List.of(Elementwise.binary(m, t(in, 0), t(in, 1), resultType(op, 0), semantics));
        }
        return switch (m) {
            case "constant" -> List.of(Tensor.of((DenseElementsAttr) attr(op, "value")));
            case "iota" -> List.of(iota(resultType(op, 0), (int) longAttr(op, "iota_dimension")));
            case "convert" -> List.of(Elementwise.convert(t(in, 0), resultType(op, 0)));
            case "bitcast_convert" -> List.of(bitcast(t(in, 0), resultType(op, 0)));
            case "compare" ->
                    List.of(
                            Elementwise.compare(
                                    enumValue(attr(op, "comparison_direction")),
                                    op.attribute("compare_type")
                                            .map(Interpreter::enumValue)
                                            .orElse(null),
                                    t(in, 0),
                                    t(in, 1),
                                    resultType(op, 0)));
            case "select" -> List.of(select(t(in, 0), t(in, 1), t(in, 2)));
            case "clamp" -> List.of(clamp(t(in, 0), t(in, 1), t(in, 2), resultType(op, 0)));
            case "reshape" -> List.of(t(in, 0).withType(resultType(op, 0)));
            case "broadcast_in_dim" ->
                    List.of(
                            broadcastInDim(
                                    t(in, 0),
                                    longs(op, "broadcast_dimensions"),
                                    resultType(op, 0)));
            case "transpose" ->
                    List.of(transpose(t(in, 0), longs(op, "permutation"), resultType(op, 0)));
            case "reverse" -> List.of(reverse(t(in, 0), longs(op, "dimensions")));
            case "slice" ->
                    List.of(
                            slice(
                                    t(in, 0),
                                    longs(op, "start_indices"),
                                    longs(op, "strides"),
                                    resultType(op, 0)));
            case "concatenate" ->
                    List.of(concatenate(in, (int) longAttr(op, "dimension"), resultType(op, 0)));
            case "pad" ->
                    List.of(
                            pad(
                                    t(in, 0),
                                    t(in, 1),
                                    longs(op, "edge_padding_low"),
                                    longs(op, "interior_padding"),
                                    resultType(op, 0)));
            case "dynamic_slice" -> List.of(dynamicSlice(in, resultType(op, 0)));
            case "dynamic_update_slice" -> List.of(dynamicUpdateSlice(in));
            case "gather" -> List.of(gather(op, t(in, 0), t(in, 1)));
            case "scatter" -> new ArrayList<>(scatter(op, in));
            case "reduce" -> new ArrayList<>(reduce(op, in));
            case "reduce_window" -> new ArrayList<>(reduceWindow(op, in));
            case "select_and_scatter" -> List.of(selectAndScatter(op, in));
            case "dot_general" -> List.of(dotGeneral(op, t(in, 0), t(in, 1)));
            case "dot" -> List.of(dot(t(in, 0), t(in, 1), resultType(op, 0)));
            case "convolution" -> List.of(Convolution.convolve(op, t(in, 0), t(in, 1), null));
            case "dynamic_conv" -> List.of(Convolution.convolve(op, t(in, 0), t(in, 1), t(in, 2)));
            case "sort" -> new ArrayList<>(sort(op, in));
            case "while" -> whileOp(op, in);
            case "if" -> region(op.region(isTrue(t(in, 0)) ? 0 : 1), List.of());
            case "case" -> {
                long index = t(in, 0).getLong(0);
                int branches = op.regions().size();
                yield region(
                        op.region(index < 0 || index >= branches ? branches - 1 : (int) index),
                        List.of());
            }
            case "map" -> List.of(map(op, in));
            case "reduce_precision" ->
                    List.of(
                            Precision.reduce(
                                    t(in, 0),
                                    (int) longAttr(op, "exponent_bits"),
                                    (int) longAttr(op, "mantissa_bits")));
            case "tuple" -> List.of(new ArrayList<>(in));
            case "get_tuple_element" ->
                    List.of(((List<?>) in.getFirst()).get((int) longAttr(op, "index")));
            case "optimization_barrier" -> in;
            case "after_all", "create_token" -> List.of(TOKEN);
            case "custom_call" -> customCall(op, in);
            default -> throw new InterpreterException("stablehlo." + m + " is not supported");
        };
    }

    // ================================================================== helpers

    static Tensor t(List<Object> in, int i) {
        Object o = in.get(i);
        if (o instanceof Tensor t) {
            return t;
        }
        throw new InterpreterException("operand #" + i + " is not a tensor: " + o);
    }

    static TensorType resultType(Operation op, int i) {
        Type type = op.result(i).type();
        if (type instanceof TensorType tt) {
            return tt;
        }
        throw new InterpreterException("result #" + i + " is not a tensor: " + type.mlir());
    }

    static Attribute attr(Operation op, String name) {
        return op.attribute(name)
                .orElseThrow(() -> new InterpreterException(op.name() + " has no " + name));
    }

    static long longAttr(Operation op, String name) {
        Attribute a = attr(op, name);
        if (a instanceof IntegerAttr i) {
            return i.value();
        }
        throw new InterpreterException(op.name() + "." + name + " is not an integer: " + a);
    }

    static long[] longs(Operation op, String name) {
        Attribute a = op.attribute(name).orElse(null);
        return switch (a) {
            case null -> null;
            case DenseArrayAttr d -> d.values();
            case DenseElementsAttr d -> {
                long[] v = new long[(int) d.numElements()];
                for (int i = 0; i < v.length; i++) {
                    v[i] = d.getLong(i);
                }
                yield v;
            }
            default -> throw new InterpreterException(op.name() + "." + name + ": " + a);
        };
    }

    private static String enumValue(Attribute a) {
        if (a instanceof EnumAttr e) {
            return e.value();
        }
        throw new InterpreterException("expected an enum, got " + a);
    }

    private static boolean isTrue(Tensor pred) {
        return pred.getLong(0) != 0;
    }

    private static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    // ================================================================== element-wise extras

    private static Tensor iota(TensorType type, int dim) {
        long[] dims = type.dims();
        int n = (int) type.numElements();
        if (type.elementType() instanceof com.qxotic.jota.stablehlo.type.ComplexType) {
            double[] re = new double[n];
            for (int i = 0; i < n; i++) {
                re[i] = Tensor.unravel(i, dims)[dim];
            }
            return Tensor.ofComplex(type, re, new double[n]);
        }
        if (type.elementType() instanceof FloatType) {
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = Tensor.unravel(i, dims)[dim];
            }
            return Tensor.ofDoubles(type, v);
        }
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = Tensor.unravel(i, dims)[dim];
        }
        return Tensor.ofLongs(type, v);
    }

    private static Tensor select(Tensor pred, Tensor onTrue, Tensor onFalse) {
        int n = onTrue.size();
        long[] b = new long[n];
        long[] m = onTrue.isComplex() ? new long[n] : null;
        for (int i = 0; i < n; i++) {
            boolean p = pred.getLong(pred.size() == 1 ? 0 : i) != 0;
            Tensor src = p ? onTrue : onFalse;
            b[i] = src.rawBits(i);
            if (m != null) {
                m[i] = src.rawImag(i);
            }
        }
        return m == null
                ? Tensor.ofBits(onTrue.type(), b)
                : Tensor.ofComplexBits(onTrue.type(), b, m);
    }

    private static Tensor clamp(Tensor min, Tensor x, Tensor max, TensorType r) {
        Tensor lo = min.size() == 1 && x.size() != 1 ? min.broadcastElement(0, x.type()) : min;
        Tensor hi = max.size() == 1 && x.size() != 1 ? max.broadcastElement(0, x.type()) : max;
        return Elementwise.binary("minimum", Elementwise.binary("maximum", x, lo, r), hi, r);
    }

    /** Reinterprets the bits; element widths may differ (little-endian regrouping). */
    private static Tensor bitcast(Tensor x, TensorType r) {
        int wIn = x.bitWidth();
        int wOut = Tensor.ofBits(r, new long[(int) r.numElements()]).bitWidth();
        if (x.isComplex()
                || r.elementType() instanceof com.qxotic.jota.stablehlo.type.ComplexType) {
            throw new InterpreterException("bitcast_convert of complex values");
        }
        int n = (int) r.numElements();
        long[] out = new long[n];
        if (wIn == wOut) {
            for (int i = 0; i < n; i++) {
                out[i] = x.rawBits(i);
            }
        } else if (wIn > wOut) { // one input splits into several outputs (last dim)
            int k = wIn / wOut;
            long mask = wOut == 64 ? -1L : (1L << wOut) - 1;
            for (int i = 0; i < n; i++) {
                out[i] = (x.rawBits(i / k) >>> (wOut * (i % k))) & mask;
            }
        } else { // several inputs (last dim) join into one output
            int k = wOut / wIn;
            long mask = wIn == 64 ? -1L : (1L << wIn) - 1;
            for (int i = 0; i < n; i++) {
                long v = 0;
                for (int j = 0; j < k; j++) {
                    v |= (x.rawBits(i * k + j) & mask) << (wIn * j);
                }
                out[i] = v;
            }
        }
        return Tensor.ofBits(r, out);
    }

    // ================================================================== shape ops

    private static Tensor broadcastInDim(Tensor x, long[] map, TensorType r) {
        long[] rd = r.dims();
        int n = (int) r.numElements();
        int[] idx = new int[n];
        long[] xd = x.dims();
        long[] xs = Tensor.strides(xd);
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            long lin = 0;
            for (int d = 0; d < map.length; d++) {
                long v = xd[d] == 1 ? 0 : ri[(int) map[d]];
                lin += v * xs[d];
            }
            idx[i] = (int) lin;
        }
        return x.select(r, idx);
    }

    private static Tensor transpose(Tensor x, long[] perm, TensorType r) {
        long[] rd = r.dims();
        long[] xs = Tensor.strides(x.dims());
        int n = (int) r.numElements();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            long lin = 0;
            for (int d = 0; d < perm.length; d++) {
                lin += ri[d] * xs[(int) perm[d]];
            }
            idx[i] = (int) lin;
        }
        return x.select(r, idx);
    }

    private static Tensor reverse(Tensor x, long[] dims) {
        long[] xd = x.dims();
        int n = x.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, xd);
            for (long d : dims) {
                ri[(int) d] = xd[(int) d] - 1 - ri[(int) d];
            }
            idx[i] = (int) Tensor.ravel(ri, xd);
        }
        return x.select(x.type(), idx);
    }

    private static Tensor slice(Tensor x, long[] start, long[] strides, TensorType r) {
        long[] rd = r.dims();
        long[] xd = x.dims();
        int n = (int) r.numElements();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            for (int d = 0; d < ri.length; d++) {
                ri[d] = start[d] + ri[d] * strides[d];
            }
            idx[i] = (int) Tensor.ravel(ri, xd);
        }
        return x.select(r, idx);
    }

    private static Tensor concatenate(List<Object> in, int dim, TensorType r) {
        long[] rd = r.dims();
        int n = (int) r.numElements();
        long[] b = new long[n];
        boolean complex = t(in, 0).isComplex();
        long[] m = complex ? new long[n] : null;
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            long pos = ri[dim];
            for (Object o : in) {
                Tensor x = (Tensor) o;
                if (pos < x.dim(dim)) {
                    ri[dim] = pos;
                    int k = (int) Tensor.ravel(ri, x.dims());
                    b[i] = x.rawBits(k);
                    if (complex) {
                        m[i] = x.rawImag(k);
                    }
                    break;
                }
                pos -= x.dim(dim);
            }
        }
        return complex ? Tensor.ofComplexBits(r, b, m) : Tensor.ofBits(r, b);
    }

    private static Tensor pad(Tensor x, Tensor value, long[] low, long[] interior, TensorType r) {
        long[] rd = r.dims();
        long[] xd = x.dims();
        int n = (int) r.numElements();
        long[] b = new long[n];
        long[] m = x.isComplex() ? new long[n] : null;
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            long[] xi = new long[ri.length];
            boolean inside = true;
            for (int d = 0; d < ri.length && inside; d++) {
                long p = ri[d] - low[d];
                long step = interior[d] + 1;
                if (p < 0 || p % step != 0 || p / step >= xd[d]) {
                    inside = false;
                } else {
                    xi[d] = p / step;
                }
            }
            if (inside) {
                int k = (int) Tensor.ravel(xi, xd);
                b[i] = x.rawBits(k);
                if (m != null) {
                    m[i] = x.rawImag(k);
                }
            } else {
                b[i] = value.rawBits(0);
                if (m != null) {
                    m[i] = value.rawImag(0);
                }
            }
        }
        return m == null ? Tensor.ofBits(r, b) : Tensor.ofComplexBits(r, b, m);
    }

    /** Start indices clamped so that the slice fits ({@code dynamic_slice} C-semantics). */
    private static long[] clampedStarts(List<Object> in, int first, long[] xd, long[] sizes) {
        long[] s = new long[xd.length];
        for (int d = 0; d < xd.length; d++) {
            long v = t(in, first + d).getLong(0);
            s[d] = Math.min(Math.max(v, 0), xd[d] - sizes[d]);
        }
        return s;
    }

    private static Tensor dynamicSlice(List<Object> in, TensorType r) {
        Tensor x = t(in, 0);
        long[] rd = r.dims();
        long[] start = clampedStarts(in, 1, x.dims(), rd);
        long[] strides = new long[rd.length];
        Arrays.fill(strides, 1);
        return slice(x, start, strides, r);
    }

    private static Tensor dynamicUpdateSlice(List<Object> in) {
        Tensor x = t(in, 0);
        Tensor u = t(in, 1);
        long[] xd = x.dims();
        long[] start = clampedStarts(in, 2, xd, u.dims());
        long[] b = x.bitsCopy();
        long[] m = x.imagCopy();
        for (int i = 0; i < u.size(); i++) {
            long[] ui = Tensor.unravel(i, u.dims());
            for (int d = 0; d < ui.length; d++) {
                ui[d] += start[d];
            }
            int k = (int) Tensor.ravel(ui, xd);
            b[k] = u.rawBits(i);
            if (m != null) {
                m[k] = u.rawImag(i);
            }
        }
        return m == null ? Tensor.ofBits(x.type(), b) : Tensor.ofComplexBits(x.type(), b, m);
    }

    // ================================================================== gather / scatter

    private static boolean contains(long[] a, long v) {
        for (long x : a) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    /** {@code start_indices[batch..., :, batch...]}: the index vector at a batch position. */
    private static long[] indexVector(Tensor indices, long[] batch, long indexVectorDim) {
        long[] id = indices.dims();
        int rank = id.length;
        if (indexVectorDim == rank) { // implicit trailing dimension of size 1
            return new long[] {indices.getLong((int) Tensor.ravel(batch, id))};
        }
        long[] pos = new long[rank];
        int b = 0;
        for (int d = 0; d < rank; d++) {
            pos[d] = d == indexVectorDim ? 0 : batch[b++];
        }
        long[] v = new long[(int) id[(int) indexVectorDim]];
        for (int k = 0; k < v.length; k++) {
            pos[(int) indexVectorDim] = k;
            v[k] = indices.getLong((int) Tensor.ravel(pos, id));
        }
        return v;
    }

    private Tensor gather(Operation op, Tensor x, Tensor indices) {
        GatherDimensionNumbers g = GatherDimensionNumbers.from(attr(op, "dimension_numbers"));
        long[] sliceSizes = longs(op, "slice_sizes");
        TensorType r = resultType(op, 0);
        long[] rd = r.dims();
        long[] xd = x.dims();
        long[] offsetDims = g.offsetDims();
        int n = (int) r.numElements();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            long[] ri = Tensor.unravel(i, rd);
            List<Long> batchL = new ArrayList<>();
            List<Long> offsetL = new ArrayList<>();
            for (int d = 0; d < rd.length; d++) {
                (contains(offsetDims, d) ? offsetL : batchL).add(ri[d]);
            }
            long[] batch = batchL.stream().mapToLong(Long::longValue).toArray();
            long[] start = indexVector(indices, batch, g.indexVectorDim());
            long[] full = new long[xd.length];
            for (int k = 0; k < g.startIndexMap().length; k++) {
                int d = (int) g.startIndexMap()[k];
                full[d] = Math.min(Math.max(start[k], 0), xd[d] - sliceSizes[d]);
            }
            for (int k = 0; k < g.operandBatchingDims().length; k++) {
                long ds = g.startIndicesBatchingDims()[k];
                full[(int) g.operandBatchingDims()[k]] +=
                        batch[(int) (ds < g.indexVectorDim() ? ds : ds - 1)];
            }
            int o = 0;
            for (int d = 0; d < xd.length; d++) {
                if (contains(g.collapsedSliceDims(), d) || contains(g.operandBatchingDims(), d)) {
                    continue;
                }
                full[d] += offsetL.get(o++);
            }
            idx[i] = (int) Tensor.ravel(full, xd);
        }
        return x.select(r, idx);
    }

    private List<Tensor> scatter(Operation op, List<Object> in) {
        int count = (in.size() - 1) / 2;
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> updates = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            inputs.add(t(in, k));
            updates.add(t(in, count + 1 + k));
        }
        Tensor indices = t(in, count);
        ScatterDimensionNumbers s =
                ScatterDimensionNumbers.from(attr(op, "scatter_dimension_numbers"));
        long[] xd = inputs.getFirst().dims();
        long[] ud = updates.getFirst().dims();
        Tensor[] results = inputs.toArray(new Tensor[0]);
        long[][] bits = new long[count][];
        long[][] imag = new long[count][];
        for (int k = 0; k < count; k++) {
            bits[k] = results[k].bitsCopy();
            imag[k] = results[k].imagCopy();
        }
        Region body = op.region(0);
        for (int u = 0; u < updates.getFirst().size(); u++) {
            long[] ui = Tensor.unravel(u, ud);
            List<Long> scatterL = new ArrayList<>();
            List<Long> windowL = new ArrayList<>();
            for (int d = 0; d < ud.length; d++) {
                (contains(s.updateWindowDims(), d) ? windowL : scatterL).add(ui[d]);
            }
            long[] scatterIndex = scatterL.stream().mapToLong(Long::longValue).toArray();
            long[] start = indexVector(indices, scatterIndex, s.indexVectorDim());
            long[] full = new long[xd.length];
            for (int k = 0; k < s.scatterDimsToOperandDims().length; k++) {
                full[(int) s.scatterDimsToOperandDims()[k]] = start[k];
            }
            for (int k = 0; k < s.inputBatchingDims().length; k++) {
                long ds = s.scatterIndicesBatchingDims()[k];
                full[(int) s.inputBatchingDims()[k]] +=
                        scatterIndex[(int) (ds < s.indexVectorDim() ? ds : ds - 1)];
            }
            int w = 0;
            for (int d = 0; d < xd.length; d++) {
                if (contains(s.insertedWindowDims(), d) || contains(s.inputBatchingDims(), d)) {
                    continue;
                }
                full[d] += windowL.get(w++);
            }
            long target = Tensor.ravel(full, xd);
            if (target < 0) {
                continue; // out of bounds: the update is skipped
            }
            List<Tensor> args = new ArrayList<>();
            for (int k = 0; k < count; k++) {
                args.add(scalarFrom(results[k], bits[k], imag[k], (int) target));
            }
            for (int k = 0; k < count; k++) {
                args.add(updates.get(k).scalar(u));
            }
            List<Tensor> combined = scalarRegion(body, args);
            for (int k = 0; k < count; k++) {
                bits[k][(int) target] = combined.get(k).rawBits(0);
                if (imag[k] != null) {
                    imag[k][(int) target] = combined.get(k).rawImag(0);
                }
            }
        }
        List<Tensor> out = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            TensorType rt = resultType(op, k);
            out.add(
                    imag[k] == null
                            ? Tensor.ofBits(rt, bits[k])
                            : Tensor.ofComplexBits(rt, bits[k], imag[k]));
        }
        return out;
    }

    private static Tensor scalarFrom(Tensor like, long[] bits, long[] imag, int i) {
        TensorType st = like.type().withShape();
        return imag == null
                ? Tensor.ofBits(st, new long[] {bits[i]})
                : Tensor.ofComplexBits(st, new long[] {bits[i]}, new long[] {imag[i]});
    }

    // ================================================================== reductions

    private List<Tensor> reduce(Operation op, List<Object> in) {
        int count = in.size() / 2;
        long[] dims = longs(op, "dimensions");
        Region body = op.region(0);
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> acc = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            inputs.add(t(in, k));
        }
        TensorType r0 = resultType(op, 0);
        int n = (int) r0.numElements();
        long[] xd = inputs.getFirst().dims();
        // result position -> list of input positions, in row-major order
        int[] owner = new int[inputs.getFirst().size()];
        long[] rd = r0.dims();
        for (int i = 0; i < owner.length; i++) {
            long[] xi = Tensor.unravel(i, xd);
            long[] ri = new long[rd.length];
            int k = 0;
            for (int d = 0; d < xd.length; d++) {
                if (!contains(dims, d)) {
                    ri[k++] = xi[d];
                }
            }
            owner[i] = (int) Tensor.ravel(ri, rd);
        }
        Tensor[][] state = new Tensor[n][count];
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < count; k++) {
                state[j][k] = t(in, count + k);
            }
        }
        for (int i = 0; i < owner.length; i++) {
            int j = owner[i];
            List<Tensor> args = new ArrayList<>(Arrays.asList(state[j]));
            for (int k = 0; k < count; k++) {
                args.add(inputs.get(k).scalar(i));
            }
            state[j] = scalarRegion(body, args).toArray(new Tensor[0]);
        }
        return assemble(op, state, count);
    }

    /** Builds the {@code count} results from per-position scalar states. */
    private static List<Tensor> assemble(Operation op, Tensor[][] state, int count) {
        List<Tensor> out = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            TensorType rt = resultType(op, k);
            int n = (int) rt.numElements();
            long[] b = new long[n];
            long[] m = null;
            for (int j = 0; j < n; j++) {
                Tensor s = state[j][k];
                b[j] = s.rawBits(0);
                if (s.isComplex()) {
                    if (m == null) {
                        m = new long[n];
                    }
                    m[j] = s.rawImag(0);
                }
            }
            out.add(m == null ? Tensor.ofBits(rt, b) : Tensor.ofComplexBits(rt, b, m));
        }
        return out;
    }

    private List<Tensor> reduceWindow(Operation op, List<Object> in) {
        int count = in.size() / 2;
        Region body = op.region(0);
        Tensor x0 = t(in, 0);
        int rank = x0.rank();
        long[] xd = x0.dims();
        long[] window = longs(op, "window_dimensions");
        long[] strides = orOnes(longs(op, "window_strides"), rank);
        long[] baseDil = orOnes(longs(op, "base_dilations"), rank);
        long[] winDil = orOnes(longs(op, "window_dilations"), rank);
        long[] padding = longs(op, "padding");
        TensorType r0 = resultType(op, 0);
        int n = (int) r0.numElements();
        long[] rd = r0.dims();
        int windowSize = (int) Arrays.stream(window).reduce(1, (a, b) -> a * b);
        Tensor[][] state = new Tensor[n][count];
        for (int j = 0; j < n; j++) {
            long[] ri = Tensor.unravel(j, rd);
            for (int k = 0; k < count; k++) {
                state[j][k] = t(in, count + k);
            }
            for (int w = 0; w < windowSize; w++) {
                long[] wi = Tensor.unravel(w, window);
                long[] xi = new long[rank];
                boolean inside = true;
                for (int d = 0; d < rank; d++) {
                    long low = padding == null ? 0 : padding[2 * d];
                    long p = ri[d] * strides[d] + wi[d] * winDil[d] - low;
                    if (p < 0 || p % baseDil[d] != 0 || p / baseDil[d] >= xd[d]) {
                        inside = false;
                        break;
                    }
                    xi[d] = p / baseDil[d];
                }
                List<Tensor> args = new ArrayList<>(Arrays.asList(state[j]));
                for (int k = 0; k < count; k++) {
                    // padding and base-dilation holes hold the init value
                    args.add(
                            inside
                                    ? t(in, k).scalar((int) Tensor.ravel(xi, xd))
                                    : t(in, count + k));
                }
                state[j] = scalarRegion(body, args).toArray(new Tensor[0]);
            }
        }
        return assemble(op, state, count);
    }

    private static long[] orOnes(long[] v, int rank) {
        if (v != null && v.length > 0) {
            return v;
        }
        long[] ones = new long[rank];
        Arrays.fill(ones, 1);
        return ones;
    }

    /**
     * {@code select_and_scatter}: in each window of the (padded) operand, {@code select} picks one
     * element; the matching {@code source} value is combined into the result there with {@code
     * scatter}, starting from {@code init_value}.
     */
    private Tensor selectAndScatter(Operation op, List<Object> in) {
        Tensor x = t(in, 0);
        Tensor source = t(in, 1);
        Tensor init = t(in, 2);
        int rank = x.rank();
        long[] xd = x.dims();
        long[] window = orOnes(longs(op, "window_dimensions"), rank);
        long[] strides = orOnes(longs(op, "window_strides"), rank);
        long[] padding = longs(op, "padding");
        Region selectRegion = op.region(0);
        Region scatterRegion = op.region(1);
        TensorType r = resultType(op, 0);
        Tensor[] result = new Tensor[x.size()];
        Arrays.fill(result, init);
        int windowSize = (int) Arrays.stream(window).reduce(1, (a, b) -> a * b);
        for (int s = 0; s < source.size(); s++) {
            long[] si = Tensor.unravel(s, source.dims());
            int selected = -1;
            for (int w = 0; w < windowSize; w++) {
                long[] wi = Tensor.unravel(w, window);
                long[] xi = new long[rank];
                boolean inside = true;
                for (int d = 0; d < rank; d++) {
                    long low = padding == null ? 0 : padding[2 * d];
                    xi[d] = si[d] * strides[d] + wi[d] - low;
                    if (xi[d] < 0 || xi[d] >= xd[d]) {
                        inside = false;
                        break;
                    }
                }
                if (!inside) {
                    continue;
                }
                int k = (int) Tensor.ravel(xi, xd);
                // padding is never selected; select(a, b) is true when a stays selected over b
                if (selected < 0
                        || !isTrue(
                                scalarRegion(selectRegion, List.of(x.scalar(selected), x.scalar(k)))
                                        .getFirst())) {
                    selected = k;
                }
            }
            if (selected >= 0) {
                result[selected] =
                        scalarRegion(scatterRegion, List.of(result[selected], source.scalar(s)))
                                .getFirst();
            }
        }
        Tensor[][] state = new Tensor[result.length][1];
        for (int i = 0; i < result.length; i++) {
            state[i][0] = result[i];
        }
        return assemble(op, state, 1).getFirst().withType(r);
    }

    private List<Tensor> sort(Operation op, List<Object> in) {
        int count = in.size();
        long dimAttr = op.attribute("dimension").map(a -> ((IntegerAttr) a).value()).orElse(-1L);
        Tensor x0 = t(in, 0);
        int rank = x0.rank();
        int dim = (int) (dimAttr < 0 ? dimAttr + rank : dimAttr);
        long[] xd = x0.dims();
        Region cmp = op.region(0);
        long[][] bits = new long[count][];
        long[][] imag = new long[count][];
        for (int k = 0; k < count; k++) {
            bits[k] = t(in, k).bitsCopy();
            imag[k] = t(in, k).imagCopy();
        }
        long len = xd[dim];
        long stride = Tensor.strides(xd)[dim];
        for (int i = 0; i < x0.size(); i++) {
            if (Tensor.unravel(i, xd)[dim] != 0) {
                continue; // one sort per slice, started at its first element
            }
            Integer[] order = new Integer[(int) len];
            for (int j = 0; j < len; j++) {
                order[j] = (int) (i + j * stride);
            }
            // merge sort is stable, as is_stable asks; an unstable sort may order equal keys
            // differently, which the spec allows
            Arrays.sort(
                    order,
                    (a, b) -> {
                        if (less(cmp, in, a, b)) {
                            return -1;
                        }
                        return less(cmp, in, b, a) ? 1 : 0;
                    });
            for (int k = 0; k < count; k++) {
                Tensor src = t(in, k);
                for (int j = 0; j < len; j++) {
                    int dst = (int) (i + j * stride);
                    bits[k][dst] = src.rawBits(order[j]);
                    if (imag[k] != null) {
                        imag[k][dst] = src.rawImag(order[j]);
                    }
                }
            }
        }
        List<Tensor> out = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            TensorType rt = resultType(op, k);
            out.add(
                    imag[k] == null
                            ? Tensor.ofBits(rt, bits[k])
                            : Tensor.ofComplexBits(rt, bits[k], imag[k]));
        }
        return out;
    }

    private boolean less(Region cmp, List<Object> in, int a, int b) {
        List<Tensor> args = new ArrayList<>();
        for (Object o : in) {
            Tensor x = (Tensor) o;
            args.add(x.scalar(a));
            args.add(x.scalar(b));
        }
        return isTrue(scalarRegion(cmp, args).getFirst());
    }

    private Tensor map(Operation op, List<Object> in) {
        TensorType r = resultType(op, 0);
        Region body = op.region(0);
        Tensor[][] state = new Tensor[(int) r.numElements()][1];
        for (int i = 0; i < state.length; i++) {
            List<Tensor> args = new ArrayList<>();
            for (Object o : in) {
                args.add(((Tensor) o).scalar(i));
            }
            state[i][0] = scalarRegion(body, args).getFirst();
        }
        return assemble(op, state, 1).getFirst();
    }

    // ================================================================== contractions

    private static Tensor dot(Tensor l, Tensor r, TensorType rt) {
        long[] lc = {l.rank() - 1};
        long[] rc = {0};
        return contract(l, r, new long[0], new long[0], lc, rc, rt);
    }

    private static Tensor dotGeneral(Operation op, Tensor l, Tensor r) {
        DotDimensionNumbers d = DotDimensionNumbers.from(attr(op, "dot_dimension_numbers"));
        return contract(
                l,
                r,
                d.lhsBatchingDimensions(),
                d.rhsBatchingDimensions(),
                d.lhsContractingDimensions(),
                d.rhsContractingDimensions(),
                resultType(op, 0));
    }

    /**
     * Result dims: batch, lhs free, rhs free. Sums in the order of the contracting index: f64 in
     * double, narrower floats in f32 (as XLA accumulates, products rounded too), integers in long;
     * complex values as two real accumulators.
     */
    private static Tensor contract(
            Tensor l, Tensor r, long[] lb, long[] rb, long[] lc, long[] rc, TensorType rt) {
        boolean complex = rt.elementType() instanceof com.qxotic.jota.stablehlo.type.ComplexType;
        long[] ld = l.dims();
        long[] rdm = r.dims();
        List<Integer> lFree = new ArrayList<>();
        for (int d = 0; d < ld.length; d++) {
            if (!contains(lb, d) && !contains(lc, d)) {
                lFree.add(d);
            }
        }
        List<Integer> rFree = new ArrayList<>();
        for (int d = 0; d < rdm.length; d++) {
            if (!contains(rb, d) && !contains(rc, d)) {
                rFree.add(d);
            }
        }
        long[] cdims = new long[lc.length];
        for (int k = 0; k < lc.length; k++) {
            cdims[k] = ld[(int) lc[k]];
        }
        int csize = (int) Arrays.stream(cdims).reduce(1, (a, b) -> a * b);
        long[] outDims = rt.dims();
        int n = (int) rt.numElements();
        boolean floatResult = rt.elementType() instanceof FloatType || complex;
        com.qxotic.jota.stablehlo.type.FloatFormat format =
                complex
                        ? ((com.qxotic.jota.stablehlo.type.ComplexType) rt.elementType())
                                .elementType()
                                .format()
                        : floatResult ? ((FloatType) rt.elementType()).format() : null;
        boolean f32 = floatResult && format != com.qxotic.jota.stablehlo.type.FloatFormat.F64;
        double[] fo = floatResult ? new double[n] : null;
        double[] fi = complex ? new double[n] : null;
        long[] io = floatResult ? null : new long[n];
        for (int i = 0; i < n; i++) {
            long[] oi = Tensor.unravel(i, outDims);
            long[] li = new long[ld.length];
            long[] ri = new long[rdm.length];
            int p = 0;
            for (int k = 0; k < lb.length; k++, p++) {
                li[(int) lb[k]] = oi[p];
                ri[(int) rb[k]] = oi[p];
            }
            for (int d : lFree) {
                li[d] = oi[p++];
            }
            for (int d : rFree) {
                ri[d] = oi[p++];
            }
            double fs = 0;
            double fsi = 0;
            long is = 0;
            for (int c = 0; c < csize; c++) {
                long[] ci = Tensor.unravel(c, cdims);
                for (int k = 0; k < lc.length; k++) {
                    li[(int) lc[k]] = ci[k];
                    ri[(int) rc[k]] = ci[k];
                }
                int a = (int) Tensor.ravel(li, ld);
                int b = (int) Tensor.ravel(ri, rdm);
                if (complex) {
                    double xr = l.getDouble(a);
                    double xi = l.isComplex() ? l.getImag(a) : 0;
                    double yr = r.getDouble(b);
                    double yi = r.isComplex() ? r.getImag(b) : 0;
                    fs = acc(fs, acc(xr * yr, -(xi * yi), f32), f32);
                    fsi = acc(fsi, acc(xr * yi, xi * yr, f32), f32);
                } else if (floatResult) {
                    double prod = l.getDouble(a) * r.getDouble(b);
                    fs = acc(fs, f32 ? (float) prod : prod, f32);
                } else {
                    is += l.getLong(a) * r.getLong(b);
                }
            }
            if (floatResult) {
                fo[i] = fs;
                if (complex) {
                    fi[i] = fsi;
                }
            } else {
                io[i] = is;
            }
        }
        if (complex) {
            return Tensor.ofComplex(rt, fo, fi);
        }
        if (floatResult) {
            return Tensor.ofDoubles(rt, fo);
        }
        if (((IntegerType) rt.elementType()).isBoolean()) {
            for (int i = 0; i < n; i++) {
                io[i] = io[i] != 0 ? 1 : 0;
            }
        }
        return Tensor.ofLongs(rt, io);
    }

    static double acc(double sum, double term, boolean f32) {
        double s = sum + term;
        return f32 ? (float) s : s;
    }

    // ================================================================== control flow

    private List<Object> whileOp(Operation op, List<Object> in) {
        List<Object> state = in;
        for (long guard = 0; ; guard++) {
            if (guard > 10_000_000) {
                throw new InterpreterException("while loop did not terminate");
            }
            Tensor cond = (Tensor) region(op.region(0), state).getFirst();
            if (!isTrue(cond)) {
                return state;
            }
            state = region(op.region(1), state);
        }
    }

    private List<Object> customCall(Operation op, List<Object> in) {
        String target = ((StringAttr) attr(op, "call_target_name")).value();
        if (target.startsWith("check.")) {
            Checks.check(target, op, in);
            return op.resultTypes().stream().map(t -> (Object) null).toList();
        }
        throw new InterpreterException("custom_call @" + target + " is not supported");
    }

    static boolean booleanAttr(Operation op, String name) {
        return op.attribute(name).orElse(null) instanceof BoolAttr b && b.value();
    }

    static boolean isToken(Type t) {
        return t instanceof TokenType || t instanceof TupleType;
    }
}

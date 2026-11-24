package com.qxotic.jota.stablehlo.verify;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.FftType;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.op.*;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FutureType;
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.QuantizedType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Return-type inference, a port of StableHLO's {@code TypeInference.cpp} (and of the {@code
 * inferReturnTypes} methods of {@code StablehloOps.cpp} / {@code Base.h}).
 *
 * <p>{@link #infer(Operation)} computes the result types an op must have from its operands,
 * attributes and regions; the op's own result types are not read, so it also serves to build
 * operations ({@link #infer(String, List, List, List, List)}). Dynamic dimensions and {@code
 * #stablehlo.bounds} propagate as in the reference: concatenation adds bounds, merging takes the
 * most specific (elementwise ops) or least specific (if / case branches) dimension.
 *
 * <p>Ops whose results are not determined by their inputs (reshape, broadcast_in_dim, iota,
 * custom_call, all_gather...) have no inference: {@link #infer} returns empty. Ops that determine
 * only the result shape (convert, dot, dot_general, convolution, uniform_quantize) return {@link
 * InferredType#shapeOnly shape-only} types.
 *
 * <p>Inference assumes the op satisfies its constraints, which {@link Verifier} checks; on an op
 * too malformed to infer anything it throws {@link TypeInferenceException}.
 */
public final class TypeInference {

    private static final long DYN = TensorType.DYNAMIC;

    private TypeInference() {}

    /** Inference failed: the op is malformed. */
    public static final class TypeInferenceException extends RuntimeException {
        TypeInferenceException(String message) {
            super(message);
        }
    }

    private static TypeInferenceException error(String message) {
        return new TypeInferenceException(message);
    }

    // ================================================================== API

    /**
     * The result types of {@code op} inferred from its operands, attributes and regions, or empty
     * when the op has no return-type inference.
     *
     * @throws TypeInferenceException when the op is too malformed to infer its result types
     */
    public static Optional<List<InferredType>> infer(Operation op) {
        try {
            return Optional.ofNullable(inferOrNull(op));
        } catch (TypeInferenceException e) {
            throw e;
        } catch (RuntimeException e) { // out-of-range dimension numbers and the like
            throw error("cannot infer result types: " + e);
        }
    }

    /** As {@link #infer(Operation)} for an operation not built yet. */
    public static Optional<List<InferredType>> infer(
            String name,
            List<Value> operands,
            List<NamedAttribute> properties,
            List<NamedAttribute> attributes,
            List<Region> regions) {
        return infer(Operation.create(name, operands, List.of(), properties, attributes, regions));
    }

    /**
     * The fully inferred result types of an operation not built yet, for builders.
     *
     * @throws IllegalArgumentException when the op has no inference or infers only shapes
     */
    public static List<Type> resultTypes(
            String name,
            List<Value> operands,
            List<NamedAttribute> properties,
            List<NamedAttribute> attributes,
            List<Region> regions) {
        List<InferredType> inferred =
                infer(name, operands, properties, attributes, regions)
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                name + " has no return-type inference"));
        List<Type> out = new ArrayList<>();
        for (InferredType t : inferred) {
            if (!t.elementTypeInferred()) {
                throw new IllegalArgumentException(
                        name + " infers only the shape of its result, not its element type");
            }
            out.add(t.type());
        }
        return out;
    }

    /**
     * Mismatches between the inferred result types of {@code op} and its actual ones, as messages
     * (empty when compatible or not inferable).
     *
     * @throws TypeInferenceException when the op is too malformed to infer its result types
     */
    public static List<String> mismatches(Operation op) {
        List<InferredType> inferred = infer(op).orElse(null);
        if (inferred == null) {
            return List.of();
        }
        List<Type> actual = op.resultTypes();
        if (inferred.size() != actual.size()) {
            return List.of(
                    "inferred "
                            + inferred.size()
                            + " result types, but the op has "
                            + actual.size()
                            + " results");
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < actual.size(); i++) {
            InferredType t = inferred.get(i);
            boolean ok =
                    op.asOp() instanceof ConstantOp
                            ? constantCompatible(t.type(), actual.get(i))
                            : t.isCompatibleWith(actual.get(i));
            if (!ok) {
                out.add(
                        "inferred type "
                                + t
                                + " is incompatible with result type "
                                + actual.get(i).mlir()
                                + (actual.size() > 1 ? " of result " + i : ""));
            }
        }
        return out;
    }

    // ================================================================== compatibility

    /**
     * {@code isCompatibleForHloTypeInference}: compatible shapes (a dynamic dimension matches any
     * size, a static size must not exceed the other side's bound) and equal element types, except
     * that quantized types only need the same storage type, range, expressed type and granularity.
     */
    public static boolean isCompatible(Type a, Type b) {
        if (a instanceof TensorType x && b instanceof TensorType y) {
            return shapeCompatibleWithBounds(x, y)
                    && elementCompatible(x.elementType(), y.elementType());
        }
        if (a instanceof TupleType x && b instanceof TupleType y) {
            return allCompatible(x.types(), y.types());
        }
        if (a instanceof FutureType x && b instanceof FutureType y) {
            return allCompatible(x.types(), y.types());
        }
        if (a instanceof ElementType x && b instanceof ElementType y) {
            return elementCompatible(x, y);
        }
        return a.equals(b);
    }

    /** Shapes compatible as in {@link #isCompatible}, element types ignored. */
    public static boolean isShapeCompatible(Type a, Type b) {
        if (a instanceof TensorType x && b instanceof TensorType y) {
            return shapeCompatibleWithBounds(x, y);
        }
        return isCompatible(a, b);
    }

    private static boolean allCompatible(List<Type> a, List<Type> b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (int i = 0; i < a.size(); i++) {
            if (!isCompatible(a.get(i), b.get(i))) {
                return false;
            }
        }
        return true;
    }

    /** {@code verifyCompatibleShapeWithBounds}. */
    static boolean shapeCompatibleWithBounds(TensorType a, TensorType b) {
        if (a.rank() != b.rank()) {
            return false;
        }
        for (int i = 0; i < a.rank(); i++) {
            if (a.dim(i) != DYN && b.dim(i) != DYN && a.dim(i) != b.dim(i)) {
                return false;
            }
        }
        return withinBounds(a, b.bounds()) && withinBounds(b, a.bounds());
    }

    private static boolean withinBounds(TensorType t, long[] bounds) {
        if (bounds == null || t.rank() == 0) {
            return true;
        }
        for (int i = 0; i < Math.min(t.rank(), bounds.length); i++) {
            if (bounds[i] != DYN && t.dim(i) != DYN && bounds[i] < t.dim(i)) {
                return false;
            }
        }
        return true;
    }

    /** {@code isCompatibleElementTypeForHloTypeInference}. */
    static boolean elementCompatible(ElementType a, ElementType b) {
        if (a instanceof QuantizedType x && b instanceof QuantizedType y) {
            return intEq(x.storageType(), y.storageType())
                    && Objects.equals(x.storageMin(), y.storageMin())
                    && Objects.equals(x.storageMax(), y.storageMax())
                    && x.expressedType().equals(y.expressedType())
                    && x.isPerAxis() == y.isPerAxis();
        }
        if (a instanceof QuantizedType || b instanceof QuantizedType) {
            return false;
        }
        if (a instanceof IntegerType x && b instanceof IntegerType y) {
            return intEq(x, y);
        }
        return a.equals(b);
    }

    /** Signless and signed integers are the same type here (StableHLO only uses signless). */
    private static boolean intEq(IntegerType a, IntegerType b) {
        return a.bitWidth() == b.bitWidth() && a.isUnsigned() == b.isUnsigned();
    }

    /** {@code ConstantOp::isCompatibleReturnTypes}: a quantized result compares by storage. */
    private static boolean constantCompatible(Type inferred, Type actual) {
        if (!(inferred instanceof TensorType i) || !(actual instanceof TensorType a)) {
            return false;
        }
        ElementType ae = a.elementType();
        if (ae instanceof QuantizedType q && !(i.elementType() instanceof QuantizedType)) {
            ae = q.storageType();
        }
        ElementType ie = i.elementType();
        return i.shape().equals(a.shape())
                && (ie.equals(ae)
                        || ie instanceof IntegerType x && ae instanceof IntegerType y && intEq(x, y)
                        || ie instanceof QuantizedType && elementCompatible(ie, ae));
    }

    // ================================================================== merging (Base.cpp)

    /**
     * {@code inferMostSpecificType}: the most precise type compatible with all of {@code types}.
     */
    public static Type mostSpecific(List<Type> types) {
        return mapOverTuples(types, true);
    }

    /** {@code inferLeastSpecificType}: a type every one of {@code types} is an instance of. */
    public static Type leastSpecific(List<Type> types) {
        return mapOverTuples(types, false);
    }

    private static Type mapOverTuples(List<Type> types, boolean most) {
        if (types.isEmpty()) {
            throw error("no types to merge");
        }
        long tuples = types.stream().filter(t -> t instanceof TupleType).count();
        if (tuples == 0) {
            return merge(types, most);
        }
        if (tuples != types.size()) {
            throw error("Mismatched type kinds: either all types must be tuples, or none");
        }
        int n = ((TupleType) types.getFirst()).types().size();
        List<Type> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Type> ith = new ArrayList<>();
            for (Type t : types) {
                List<Type> elems = ((TupleType) t).types();
                if (elems.size() != n) {
                    throw error("Mismatched tuple sizes: all tuple sizes must be the same");
                }
                ith.add(elems.get(i));
            }
            out.add(mapOverTuples(ith, most));
        }
        return new TupleType(out);
    }

    private static Type merge(List<Type> types, boolean most) {
        List<TensorType> tensors = new ArrayList<>();
        for (Type t : types) {
            if (t instanceof TensorType tt) {
                tensors.add(tt);
            } else if (!most) {
                return t;
            }
        }
        if (tensors.isEmpty()) {
            return types.getFirst();
        }
        TensorType first = tensors.getFirst();
        int rank = first.rank();
        long[] sizes = first.dims();
        long[] firstBounds = first.bounds();
        long[] bounds = firstBounds != null ? firstBounds.clone() : filled(rank);
        boolean anyBounds = firstBounds != null;
        for (int k = 1; k < tensors.size(); k++) {
            TensorType t = tensors.get(k);
            if (t.rank() != rank) {
                throw error("Mismatched ranks of types " + rank + " vs " + t.rank());
            }
            long[] b = t.bounds();
            anyBounds |= b != null;
            for (int d = 0; d < rank; d++) {
                long rb = b == null ? DYN : b[d];
                long[] sb =
                        most
                                ? mostSpecificDim(d, sizes[d], t.dim(d), bounds[d], rb)
                                : leastSpecificDim(d, sizes[d], t.dim(d), bounds[d], rb);
                sizes[d] = sb[0];
                bounds[d] = sb[1];
            }
        }
        return tensor(sizes, first.elementType(), anyBounds ? bounds : null, first.encoding());
    }

    /** {@code inferMostSpecificDimAndBound}. */
    static long[] mostSpecificDim(int dim, long l, long r, long lb, long rb) {
        long size = DYN;
        long bound = DYN;
        if (l != DYN || r != DYN) {
            if (l != DYN && r != DYN && l != r) {
                throw error(
                        "Mismatched dimension sizes " + l + " and " + r + " in dimension " + dim);
            }
            size = l != DYN ? l : r;
            if (lb != DYN || rb != DYN) {
                long check = lb != DYN ? lb : rb;
                if (size > check) {
                    throw error(
                            "Mismatched dimension size "
                                    + size
                                    + " and bound "
                                    + check
                                    + " in dimension "
                                    + dim);
                }
            }
        } else if (lb != DYN && rb != DYN) {
            bound = Math.min(lb, rb);
        } else {
            bound = lb != DYN ? lb : rb;
        }
        return new long[] {size, bound};
    }

    /** {@code inferLeastSpecificDimAndBound}. */
    static long[] leastSpecificDim(int dim, long l, long r, long lb, long rb) {
        long size = DYN;
        long bound = DYN;
        if (l != DYN || r != DYN) {
            if (l != DYN && r != DYN) {
                if (l != r) {
                    throw error(
                            "Mismatched dimension sizes "
                                    + l
                                    + " and "
                                    + r
                                    + " in dimension "
                                    + dim);
                }
                size = l;
            } else if (lb != DYN || rb != DYN) {
                bound = l != DYN ? Math.max(l, rb) : Math.max(r, lb);
            }
        } else if (lb != DYN && rb != DYN) {
            bound = Math.max(lb, rb);
        }
        return new long[] {size, bound};
    }

    /** {@code inferConcatenatedDimAndBound}. */
    static long[] concatenatedDim(long l, long r, long lb, long rb) {
        if (l != DYN && r != DYN) {
            return new long[] {l + r, DYN};
        }
        long ls = l != DYN ? l : lb;
        long rs = r != DYN ? r : rb;
        return new long[] {DYN, ls != DYN && rs != DYN ? ls + rs : DYN};
    }

    // ================================================================== type construction

    private static long[] filled(int n) {
        long[] a = new long[n];
        Arrays.fill(a, DYN);
        return a;
    }

    /**
     * {@code RankedTensorType::get(sizes, element, boundsToEncoding(prototype, bounds))}: null
     * bounds keep the prototype encoding, all-dynamic bounds drop it.
     */
    private static TensorType tensor(
            long[] sizes, ElementType element, long[] bounds, Attribute prototype) {
        Attribute encoding = bounds == null ? prototype : TensorType.boundsEncoding(bounds);
        return new TensorType(boxed(sizes), element, encoding);
    }

    private static TensorType tensor(long[] sizes, ElementType element) {
        return new TensorType(boxed(sizes), element, null);
    }

    private static List<Long> boxed(long[] v) {
        List<Long> out = new ArrayList<>(v.length);
        for (long x : v) {
            out.add(x);
        }
        return out;
    }

    /** {@code getSameShapeTensorType}: same shape and encoding, new element type. */
    private static TensorType sameShape(TensorType t, ElementType element) {
        return t.withElementType(element);
    }

    private static ElementType realOf(ElementType e) {
        return e instanceof ComplexType c ? c.elementType() : e;
    }

    // ================================================================== inference per op

    private static final Set<String> SAME_AS_OPERANDS =
            Set.of(
                    "stablehlo.cbrt",
                    "stablehlo.ceil",
                    "stablehlo.count_leading_zeros",
                    "stablehlo.cosine",
                    "stablehlo.exponential",
                    "stablehlo.exponential_minus_one",
                    "stablehlo.exp2",
                    "stablehlo.floor",
                    "stablehlo.log",
                    "stablehlo.log_plus_one",
                    "stablehlo.log2",
                    "stablehlo.logistic",
                    "stablehlo.not",
                    "stablehlo.negate",
                    "stablehlo.popcnt",
                    "stablehlo.round_nearest_afz",
                    "stablehlo.round_nearest_even",
                    "stablehlo.rsqrt",
                    "stablehlo.sign",
                    "stablehlo.sine",
                    "stablehlo.sqrt",
                    "stablehlo.tan",
                    "stablehlo.tanh",
                    "stablehlo.reduce_precision",
                    "stablehlo.add",
                    "stablehlo.atan2",
                    "stablehlo.divide",
                    "stablehlo.maximum",
                    "stablehlo.minimum",
                    "stablehlo.multiply",
                    "stablehlo.power",
                    "stablehlo.remainder",
                    "stablehlo.shift_left",
                    "stablehlo.shift_right_arithmetic",
                    "stablehlo.shift_right_logical",
                    "stablehlo.subtract",
                    "stablehlo.and",
                    "stablehlo.or",
                    "stablehlo.xor",
                    "stablehlo.fma");

    private static List<InferredType> exact(Type... types) {
        List<InferredType> out = new ArrayList<>();
        for (Type t : types) {
            out.add(InferredType.of(t));
        }
        return out;
    }

    private static List<InferredType> exact(List<Type> types) {
        return types.stream().map(InferredType::of).toList();
    }

    private static TensorType tt(Value v) {
        if (v.type() instanceof TensorType t) {
            return t;
        }
        throw error("expected a ranked tensor, got " + v.type().mlir());
    }

    private static List<InferredType> inferOrNull(Operation operation) {
        if (SAME_AS_OPERANDS.contains(operation.name())) {
            if (operation.numOperands() == 0) {
                throw error("expected operands");
            }
            return exact(mostSpecific(operation.operandTypes()));
        }
        return switch (operation.asOp()) {
            case AbsOp op -> {
                TensorType t = tt(op.operand());
                yield exact(sameShape(t, realOf(t.elementType())));
            }
            case RealOp op ->
                    exact(sameShape(tt(op.operand()), realOf(tt(op.operand()).elementType())));
            case ImagOp op ->
                    exact(sameShape(tt(op.operand()), realOf(tt(op.operand()).elementType())));
            case IsFiniteOp op -> exact(sameShape(tt(op.operand()), IntegerType.I1));
            case ComplexOp op -> {
                TensorType t = tt(op.lhs());
                if (!(t.elementType() instanceof FloatType f)) {
                    throw error("complex needs floating-point operands, got " + t.mlir());
                }
                yield exact(sameShape(t, new ComplexType(f)));
            }
            case ConvertOp op -> List.of(InferredType.shapeOnly(tt(op.operand())));
            case UniformQuantizeOp op -> List.of(InferredType.shapeOnly(tt(op.operand())));
            case UniformDequantizeOp op -> {
                TensorType t = tt(op.operand());
                if (!(t.elementType() instanceof QuantizedType q)) {
                    throw error("uniform_dequantize needs a quantized operand, got " + t.mlir());
                }
                yield exact(tensor(t.dims(), q.expressedType()));
            }
            case CompareOp op -> {
                TensorType t = tt(op.lhs());
                yield exact(sameShape(t, IntegerType.I1));
            }
            case SelectOp op ->
                    exact(mostSpecific(List.of(op.onTrue().type(), op.onFalse().type())));
            case ClampOp op -> exact(op.operand().type());
            case ConstantOp op -> {
                if (!(op.value() instanceof DenseElementsAttr d)) {
                    yield null; // dense_resource and the like: no type at hand
                }
                yield exact(d.type());
            }
            case CreateTokenOp ignored -> exact(TokenType.INSTANCE);
            case AfterAllOp ignored -> exact(TokenType.INSTANCE);
            case OutfeedOp ignored -> exact(TokenType.INSTANCE);
            case SendOp ignored -> exact(TokenType.INSTANCE);
            case ReplicaIdOp ignored -> exact(tensor(new long[0], IntegerType.unsigned(32)));
            case PartitionIdOp ignored -> exact(tensor(new long[0], IntegerType.unsigned(32)));
            case GetDimensionSizeOp op -> {
                checkDim(tt(op.operand()), op.dimension());
                yield exact(tensor(new long[0], IntegerType.I32));
            }
            case TupleOp op -> exact(new TupleType(op.operation().operandTypes()));
            case GetTupleElementOp op -> {
                if (!(op.operand().type() instanceof TupleType t)) {
                    throw error("operand must be a tuple, got " + op.operand().type().mlir());
                }
                if (op.index() < 0 || op.index() >= t.types().size()) {
                    throw error("index " + op.index() + " is out of bounds of " + t.mlir());
                }
                yield exact(t.types().get((int) op.index()));
            }
            case OptimizationBarrierOp op -> exact(op.operation().operandTypes());
            case WhileOp op -> exact(op.operation().operandTypes());
            case CollectivePermuteOp op -> exact(op.operation().operandTypes());
            case IfOp op -> conditional(List.of(op.trueBranch(), op.falseBranch()));
            case CaseOp op -> conditional(op.branches());
            case AsyncStartOp op -> asyncStart(op.operation());
            case AsyncDoneOp op -> {
                if (!(op.future().type() instanceof FutureType f) || f.types().size() != 1) {
                    throw error(
                            "operand must be a future of one type, got "
                                    + op.future().type().mlir());
                }
                yield exact(f.types().getFirst());
            }
            case CollectiveBroadcastOp op ->
                    exact(dataOperands(op.operation()).stream().map(Value::type).toList());
            case CollectiveReduceOp op -> {
                List<InferredType> out = new ArrayList<>();
                for (Value v : dataOperands(op.operation())) {
                    out.add(
                            InferredType.of(
                                    sameShape(tt(v), accumulator(op.computation(), 0, tt(v)))));
                }
                yield out;
            }
            case AllReduceOp op -> {
                List<InferredType> out = new ArrayList<>();
                for (Value v : op.operands()) {
                    out.add(
                            InferredType.of(
                                    sameShape(tt(v), accumulator(op.computation(), 0, tt(v)))));
                }
                yield out;
            }
            case AllToAllOp op -> allToAll(op);
            case ReduceOp op -> reduce(op);
            case ReduceWindowOp op -> reduceWindow(op);
            case ScatterOp op -> {
                List<InferredType> out = new ArrayList<>();
                List<Value> inputs = op.inputs();
                for (int i = 0; i < inputs.size(); i++) {
                    out.add(
                            InferredType.of(
                                    sameShape(
                                            tt(inputs.get(i)),
                                            accumulator(
                                                    op.updateComputation(),
                                                    i,
                                                    tt(inputs.get(i))))));
                }
                yield out;
            }
            case SelectAndScatterOp op -> {
                TensorType t = tt(op.operand());
                yield exact(sameShape(t, accumulator(op.scatter(), 0, t)));
            }
            case SortOp op -> exact(op.operation().operandTypes());
            case MapOp op -> map(op);
            case DynamicSliceOp op -> {
                TensorType t = tt(op.operand());
                long[] sizes = op.sliceSizes();
                if (sizes.length != t.rank() || op.startIndices().size() != t.rank()) {
                    throw error(
                            "slice_sizes and start_indices must match the operand rank "
                                    + t.rank());
                }
                yield exact(tensor(sizes, t.elementType()));
            }
            case DynamicUpdateSliceOp op -> exact(op.operand().type());
            case BatchNormInferenceOp op -> exact(op.operand().type());
            case BatchNormTrainingOp op -> batchNorm(tt(op.operand()), op.featureIndex());
            case BatchNormGradOp op -> batchNorm(tt(op.operand()), op.featureIndex());
            case BroadcastOp op -> {
                TensorType t = tt(op.operand());
                long[] sizes = op.broadcastSizes();
                for (long s : sizes) {
                    if (s < 0) {
                        throw error("Broadcast with negative dimension size " + s);
                    }
                }
                long[] shape = Arrays.copyOf(sizes, sizes.length + t.rank());
                System.arraycopy(t.dims(), 0, shape, sizes.length, t.rank());
                yield exact(tensor(shape, t.elementType()));
            }
            case CholeskyOp op -> exact(op.a().type());
            case TriangularSolveOp op -> exact(op.b().type());
            case ReverseOp op -> exact(op.operand().type());
            case ConcatenateOp op -> concatenate(op);
            case TransposeOp op -> transpose(op);
            case PadOp op -> pad(op);
            case SliceOp op -> slice(op);
            case DotOp op -> dot(op);
            case DotGeneralOp op -> dotGeneral(op);
            case ConvolutionOp op -> convolution(op);
            case FftOp op -> fft(op);
            case GatherOp op -> {
                long[] sizes = op.sliceSizes();
                yield gather(
                        op.operand(),
                        op.startIndices(),
                        op.dimensionNumbers(),
                        i -> sizes[(int) i]);
            }
            case DynamicGatherOp op -> {
                TensorType ss = tt(op.sliceSizes());
                if (ss.rank() != 1 || ss.dim(0) != DYN && ss.dim(0) != tt(op.operand()).rank()) {
                    throw error(
                            "slice_sizes size ("
                                    + (ss.rank() == 1 ? ss.dim(0) : ss.rank())
                                    + ") not equal to operand rank ("
                                    + tt(op.operand()).rank()
                                    + ")");
                }
                long[] sizes = constant(op.sliceSizes());
                yield gather(
                        op.operand(),
                        op.startIndices(),
                        op.dimensionNumbers(),
                        i -> sizes == null ? DYN : sizes[(int) i]);
            }
            case SetDimensionSizeOp op -> setDimensionSize(op);
            case RngOp op -> rng(op);
            default -> null;
        };
    }

    private static void checkDim(TensorType t, long dim) {
        if (dim < 0 || dim >= t.rank()) {
            throw error(
                    "requires dimension attribute in range [0, "
                            + t.rank()
                            + "); found ("
                            + dim
                            + ")");
        }
    }

    /** Operands of collective_broadcast / collective_reduce without the dynamic root. */
    private static List<Value> dataOperands(Operation op) {
        List<Value> all = op.operands();
        return op.attribute("has_dynamic_root").isPresent() && !all.isEmpty()
                ? all.subList(0, all.size() - 1)
                : all;
    }

    private static List<Type> returned(Region region) {
        if (region.blocks().size() != 1 || region.block().terminator() == null) {
            return null;
        }
        return region.block().terminator().operandTypes();
    }

    /**
     * {@code getAccumulatorTypesOrInputTypes}: the element type of result {@code i} of a reducer
     * body, or of {@code input} when the region is empty.
     */
    private static ElementType accumulator(Region region, int i, TensorType input) {
        if (region.isEmpty()) {
            return input.elementType();
        }
        List<Type> rets = returned(region);
        if (rets == null || i >= rets.size() || !(rets.get(i) instanceof TensorType t)) {
            throw error("the body must return a tensor for each input");
        }
        return t.elementType();
    }

    /** {@code inferConditionalOp}: least specific type of each branch result. */
    private static List<InferredType> conditional(List<Region> branches) {
        if (branches.isEmpty()) {
            throw error("expect at least one branch");
        }
        List<List<Type>> rets = new ArrayList<>();
        for (Region r : branches) {
            List<Type> t = returned(r);
            if (t == null) {
                throw error("branches must have a single block with a terminator");
            }
            if (!rets.isEmpty() && t.size() != rets.getFirst().size()) {
                throw error("branches return different numbers of values");
            }
            rets.add(t);
        }
        List<InferredType> out = new ArrayList<>();
        for (int i = 0; i < rets.getFirst().size(); i++) {
            List<Type> ith = new ArrayList<>();
            for (List<Type> r : rets) {
                ith.add(r.get(i));
            }
            out.add(InferredType.of(leastSpecific(ith)));
        }
        return out;
    }

    private static List<InferredType> asyncStart(Operation op) {
        if (op.regions().isEmpty()) {
            return null; // called_computation form: the callee is elsewhere
        }
        List<Type> rets = returned(op.region(0));
        if (rets == null || rets.size() != 1) {
            throw error("'stablehlo.async_start' op region must return exactly one value");
        }
        return exact(new FutureType(List.of(rets.getFirst())));
    }

    private static List<InferredType> allToAll(AllToAllOp op) {
        long split = op.splitDimension();
        long concat = op.concatDimension();
        long count = op.splitCount();
        List<InferredType> out = new ArrayList<>();
        for (Value v : op.operands()) {
            TensorType t = tt(v);
            if (split < 0 || split >= t.rank() || concat < 0 || concat >= t.rank()) {
                throw error("split/concat dimension out of bounds for rank " + t.rank());
            }
            if (count <= 0) {
                throw error("split_count must be positive");
            }
            long[] shape = t.dims();
            int s = (int) split;
            int c = (int) concat;
            if (shape[s] != DYN && shape[s] % count != 0) {
                throw error(
                        "split dimension has size "
                                + shape[s]
                                + ", expected to be a multiple of split_count "
                                + count);
            }
            if (shape[s] != DYN) {
                shape[s] /= count;
            }
            if (shape[c] != DYN) {
                shape[c] *= count;
            }
            long[] bounds = t.bounds();
            if (bounds != null) {
                bounds = bounds.clone();
                if (bounds[s] != DYN) {
                    bounds[s] /= count;
                }
                if (bounds[c] != DYN) {
                    bounds[c] *= count;
                }
            }
            out.add(InferredType.of(tensor(shape, t.elementType(), bounds, t.encoding())));
        }
        return out;
    }

    private static List<InferredType> reduce(ReduceOp op) {
        List<Value> inputs = op.inputs();
        if (inputs.isEmpty()) {
            throw error("reduce needs inputs");
        }
        TensorType witness = tt(inputs.getFirst());
        long[] dims = op.dimensions();
        boolean[] reduced = new boolean[witness.rank()];
        for (long d : dims) {
            if (d < 0 || d >= witness.rank()) {
                throw error(
                        "Out-of-bounds dimension "
                                + d
                                + ", expected to be in range [0, "
                                + witness.rank()
                                + ")");
            }
            if (reduced[(int) d]) {
                throw error("Duplicate reduction dimension: " + d);
            }
            reduced[(int) d] = true;
        }
        long[] inBounds = witness.bounds();
        List<Long> shape = new ArrayList<>();
        List<Long> bounds = new ArrayList<>();
        for (int i = 0; i < witness.rank(); i++) {
            if (!reduced[i]) {
                shape.add(witness.dim(i));
                if (inBounds != null) {
                    bounds.add(inBounds[i]);
                }
            }
        }
        Attribute encoding =
                bounds.isEmpty()
                        ? null
                        : TensorType.boundsEncoding(
                                bounds.stream().mapToLong(Long::longValue).toArray());
        List<InferredType> out = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            out.add(
                    InferredType.of(
                            new TensorType(
                                    shape,
                                    accumulator(op.body(), i, tt(inputs.get(i))),
                                    encoding)));
        }
        return out;
    }

    /** {@code dilatedBound}. */
    private static long dilated(long bound, long dilation) {
        return bound == 0 ? 0 : (bound - 1) * dilation + 1;
    }

    /** {@code stridedBound}. */
    private static long strided(long bound, long window, long stride) {
        if (bound <= 0 || window > bound) {
            return 0;
        }
        return (bound - window) / stride + 1;
    }

    /** {@code inferWindowOutputShape} with the window attributes (empty arrays mean defaults). */
    private static long[] windowOutput(
            long[] base,
            long[] window,
            long[] strides,
            long[] baseDilations,
            long[] windowDilations,
            long[][] padding) {
        int n = window.length;
        if (base.length != n) {
            throw error("window has " + n + " dimensions, the input " + base.length);
        }
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            long stride = attr(strides, i, 1, "window-strides", n);
            long bd = attr(baseDilations, i, 1, "base-dilation factors", n);
            long wd = attr(windowDilations, i, 1, "window-dilation factors", n);
            if (window[i] != DYN && window[i] <= 0 || stride <= 0 || bd <= 0 || wd <= 0) {
                throw error("window sizes, strides and dilations must be positive");
            }
            if (base[i] == DYN || window[i] == DYN) {
                out[i] = DYN;
                continue;
            }
            long lo = padding == null ? 0 : padding[i][0];
            long hi = padding == null ? 0 : padding[i][1];
            out[i] = strided(lo + dilated(base[i], bd) + hi, dilated(window[i], wd), stride);
        }
        return out;
    }

    private static long attr(long[] values, int i, long dflt, String name, int n) {
        if (values == null || values.length == 0) {
            return dflt;
        }
        if (values.length != n) {
            throw error(
                    "expects "
                            + name
                            + " to have same dimension-size as size of window dimensions ("
                            + n
                            + "), but got: "
                            + values.length
                            + ".");
        }
        return values[i];
    }

    /** A {@code [n, 2]} padding attribute as pairs, or null. */
    private static long[][] padding(DenseElementsAttr pad, int n) {
        if (pad == null || pad.numElements() == 0) {
            return null;
        }
        if (pad.type().rank() != 2 || pad.type().dim(0) != n || pad.type().dim(1) != 2) {
            throw error("padding must have shape [" + n + ", 2], got " + pad.type().mlir());
        }
        long[][] out = new long[n][2];
        for (int i = 0; i < n; i++) {
            out[i][0] = pad.getLong(2L * i);
            out[i][1] = pad.getLong(2L * i + 1);
        }
        return out;
    }

    private static List<InferredType> reduceWindow(ReduceWindowOp op) {
        long[] window = op.windowDimensions();
        long[] strides = op.windowStrides().orElse(null);
        long[] bd = op.baseDilations().orElse(null);
        long[] wd = op.windowDilations().orElse(null);
        long[][] pad = padding(op.padding().orElse(null), window.length);
        List<InferredType> out = new ArrayList<>();
        List<Value> inputs = op.inputs();
        for (int i = 0; i < inputs.size(); i++) {
            TensorType t = tt(inputs.get(i));
            long[] shape = windowOutput(t.dims(), window, strides, bd, wd, pad);
            long[] bounds = t.bounds();
            ElementType e = accumulator(op.body(), i, t);
            if (bounds == null) {
                out.add(InferredType.of(tensor(shape, e)));
            } else {
                long[] rb = windowOutput(bounds, window, strides, bd, wd, pad);
                out.add(InferredType.of(tensor(shape, e, rb, t.encoding())));
            }
        }
        return out;
    }

    private static List<InferredType> map(MapOp op) {
        List<Type> rets = returned(op.computation());
        if (rets == null || rets.size() != 1 || !(rets.getFirst() instanceof TensorType r)) {
            throw error("computation must return a single tensor");
        }
        if (op.inputs().isEmpty()) {
            throw error("map needs inputs");
        }
        TensorType last = tt(op.inputs().getLast());
        return exact(tensor(last.dims(), r.elementType()));
    }

    private static List<InferredType> batchNorm(TensorType operand, long feature) {
        checkDim(operand, feature);
        int f = (int) feature;
        long[] bounds = operand.bounds();
        TensorType single =
                bounds == null
                        ? tensor(new long[] {operand.dim(f)}, operand.elementType())
                        : tensor(
                                new long[] {operand.dim(f)},
                                operand.elementType(),
                                new long[] {bounds[f]},
                                null);
        return exact(operand, single, single);
    }

    private static List<InferredType> concatenate(ConcatenateOp op) {
        List<Value> inputs = op.inputs();
        if (inputs.isEmpty()) {
            throw error("concatenate needs inputs");
        }
        TensorType witness = tt(inputs.getFirst());
        int rank = witness.rank();
        long dimension = op.dimension();
        if (rank == 0) {
            throw error("rank-0 values cannot be concatenated");
        }
        if (dimension < 0 || dimension >= rank) {
            throw error("dimension " + dimension + " is out-of-bounds for input rank " + rank);
        }
        long[] sizes = filled(rank);
        long[] bounds = filled(rank);
        sizes[(int) dimension] = 0;
        boolean anyBounds = false;
        Attribute witnessEncoding = null;
        for (Value v : inputs) {
            TensorType t = tt(v);
            if (t.rank() != rank) {
                throw error("operands do not match rank");
            }
            long[] b = t.bounds();
            anyBounds |= b != null;
            if (witnessEncoding == null) {
                witnessEncoding = t.encoding();
            }
            for (int d = 0; d < rank; d++) {
                long rb = b == null ? DYN : b[d];
                long[] sb =
                        d == dimension
                                ? concatenatedDim(sizes[d], t.dim(d), bounds[d], rb)
                                : mostSpecificDim(d, sizes[d], t.dim(d), bounds[d], rb);
                sizes[d] = sb[0];
                bounds[d] = sb[1];
            }
        }
        return exact(
                tensor(sizes, witness.elementType(), anyBounds ? bounds : null, witnessEncoding));
    }

    private static List<InferredType> transpose(TransposeOp op) {
        TensorType t = tt(op.operand());
        long[] perm = op.permutation();
        if (perm.length != t.rank()) {
            throw error(
                    "TransposeOp operand rank "
                            + t.rank()
                            + " does not match permutation size "
                            + perm.length);
        }
        long[] sorted = perm.clone();
        Arrays.sort(sorted);
        for (int i = 0; i < sorted.length; i++) {
            if (sorted[i] != i) {
                throw error("attribute permutation must be a permutation of [0, " + t.rank() + ")");
            }
        }
        long[] in = t.bounds();
        long[] shape = new long[perm.length];
        long[] bounds = in == null ? null : new long[perm.length];
        for (int i = 0; i < perm.length; i++) {
            shape[i] = t.dim((int) perm[i]);
            if (bounds != null) {
                bounds[i] = in[(int) perm[i]];
            }
        }
        return exact(tensor(shape, t.elementType(), bounds, t.encoding()));
    }

    private static List<InferredType> pad(PadOp op) {
        TensorType t = tt(op.operand());
        int rank = t.rank();
        long[] lo = op.edgePaddingLow();
        long[] hi = op.edgePaddingHigh();
        long[] in = op.interiorPadding();
        if (lo.length != rank || hi.length != rank || in.length != rank) {
            throw error("padding sizes must match the operand rank (" + rank + ")");
        }
        long[] inBounds = t.bounds();
        long[] shape = filled(rank);
        long[] bounds = inBounds == null ? null : filled(rank);
        for (int i = 0; i < rank; i++) {
            if (in[i] < 0) {
                throw error("Interior padding cannot be negative: " + in[i]);
            }
            boolean isStatic = t.dim(i) != DYN;
            boolean staticBound = inBounds != null && inBounds[i] != DYN;
            if (isStatic || staticBound) {
                long n = isStatic ? t.dim(i) : inBounds[i];
                long r = n + lo[i] + hi[i] + Math.max(n - 1, 0) * in[i];
                if (r < 0) {
                    throw error(
                            "Padding result in negative "
                                    + (isStatic ? "size" : "bound")
                                    + " for dimension "
                                    + i);
                }
                (isStatic ? shape : bounds)[i] = r;
            }
        }
        return exact(tensor(shape, t.elementType(), bounds, t.encoding()));
    }

    private static List<InferredType> slice(SliceOp op) {
        TensorType t = tt(op.operand());
        int rank = t.rank();
        long[] start = op.startIndices();
        long[] limit = op.limitIndices();
        long[] strides = op.strides();
        if (start.length != rank || limit.length != rank || strides.length != rank) {
            throw error(
                    "the number of start_indices, limit_indices and strides must match the rank of"
                            + " the operand ("
                            + rank
                            + ")");
        }
        long[] inBounds = t.bounds();
        long[] shape = filled(rank);
        for (int i = 0; i < rank; i++) {
            if (start[i] < 0) {
                throw error("negative start index " + start[i] + " in dimension " + i);
            }
            boolean isStatic = t.dim(i) != DYN;
            boolean staticBound = inBounds != null && inBounds[i] != DYN;
            if (isStatic || staticBound) {
                long n = isStatic ? t.dim(i) : inBounds[i];
                if (limit[i] > n) {
                    throw error(
                            "limit index "
                                    + limit[i]
                                    + " is larger than dimension "
                                    + (isStatic ? "size " : "bound ")
                                    + n
                                    + " in dimension "
                                    + i);
                }
            }
            if (start[i] > limit[i]) {
                throw error(
                        "start index "
                                + start[i]
                                + " is larger than limit index "
                                + limit[i]
                                + " in dimension "
                                + i);
            }
            if (strides[i] <= 0) {
                throw error("stride must be positive but got " + strides[i] + " in dimension " + i);
            }
            shape[i] = Math.ceilDiv(limit[i] - start[i], strides[i]);
        }
        return exact(
                tensor(
                        shape,
                        t.elementType(),
                        inBounds == null ? null : filled(rank),
                        t.encoding()));
    }

    private static List<InferredType> dot(DotOp op) {
        TensorType l = tt(op.lhs());
        TensorType r = tt(op.rhs());
        long[] lb = l.bounds();
        long[] rb = r.bounds();
        List<Long> shape = new ArrayList<>();
        List<Long> bounds = new ArrayList<>();
        int lr = l.rank();
        int rr = r.rank();
        if (lr == 1 && rr == 1 && Verifier.dimEq(l.dim(0), r.dim(0))) {
            // scalar
        } else if (lr == 2 && rr == 1 && Verifier.dimEq(l.dim(1), r.dim(0))) {
            push(shape, bounds, l, lb, 0);
        } else if (lr == 1 && rr == 2 && Verifier.dimEq(l.dim(0), r.dim(0))) {
            push(shape, bounds, r, rb, 1);
        } else if (lr == 2 && rr == 2 && Verifier.dimEq(l.dim(1), r.dim(0))) {
            push(shape, bounds, l, lb, 0);
            push(shape, bounds, r, rb, 1);
        } else {
            throw error("expected both lhs/rhs ranks to be either 1 or 2");
        }
        long[] b = bounds.stream().mapToLong(Long::longValue).toArray();
        Attribute proto = l.encoding() != null ? l.encoding() : r.encoding();
        return List.of(
                InferredType.shapeOnly(
                        tensor(
                                shape.stream().mapToLong(Long::longValue).toArray(),
                                l.elementType(),
                                b,
                                proto)));
    }

    private static void push(List<Long> shape, List<Long> bounds, TensorType t, long[] tb, int d) {
        shape.add(t.dim(d));
        bounds.add(tb == null ? DYN : tb[d]);
    }

    private static boolean contains(long[] values, long v) {
        for (long x : values) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    private static List<InferredType> dotGeneral(DotGeneralOp op) {
        TensorType l = tt(op.lhs());
        TensorType r = tt(op.rhs());
        DotDimensionNumbers dn = op.dotDimensionNumbers();
        long[] lbd = dn.lhsBatchingDimensions();
        long[] rbd = dn.rhsBatchingDimensions();
        long[] lcd = dn.lhsContractingDimensions();
        long[] rcd = dn.rhsContractingDimensions();
        if (lbd.length != rbd.length || lcd.length != rcd.length) {
            throw error(
                    "lhs and rhs should have the same number of batching and contracting"
                            + " dimensions");
        }
        long[] lb = l.bounds();
        long[] rb = r.bounds();
        List<Long> shape = new ArrayList<>();
        List<Long> bounds = new ArrayList<>();
        for (int i = 0; i < lbd.length; i++) {
            int ld = (int) lbd[i];
            int rd = (int) rbd[i];
            long[] sb =
                    mostSpecificDim(
                            i,
                            l.dim(ld),
                            r.dim(rd),
                            lb == null ? DYN : lb[ld],
                            rb == null ? DYN : rb[rd]);
            shape.add(sb[0]);
            bounds.add(sb[1]);
        }
        for (int i = 0; i < l.rank(); i++) {
            if (!contains(lbd, i) && !contains(lcd, i)) {
                push(shape, bounds, l, lb, i);
            }
        }
        for (int i = 0; i < r.rank(); i++) {
            if (!contains(rbd, i) && !contains(rcd, i)) {
                push(shape, bounds, r, rb, i);
            }
        }
        Attribute proto = l.encoding() != null ? l.encoding() : r.encoding();
        return List.of(
                InferredType.shapeOnly(
                        tensor(
                                shape.stream().mapToLong(Long::longValue).toArray(),
                                l.elementType(),
                                bounds.stream().mapToLong(Long::longValue).toArray(),
                                proto)));
    }

    private static List<InferredType> convolution(ConvolutionOp op) {
        TensorType l = tt(op.lhs());
        TensorType r = tt(op.rhs());
        ConvDimensionNumbers dn = op.dimensionNumbers();
        int rank = l.rank();
        long[] inSpatial = dn.inputSpatialDimensions();
        long[] kSpatial = dn.kernelSpatialDimensions();
        long[] outSpatial = dn.outputSpatialDimensions();
        if (rank < 2 || r.rank() != rank || inSpatial.length + 2 != rank) {
            throw error(
                    "expects convolution arguments to have "
                            + (inSpatial.length + 2)
                            + " dimensions");
        }
        long[] window = new long[kSpatial.length];
        long[] base = new long[inSpatial.length];
        for (int i = 0; i < window.length; i++) {
            window[i] = r.dim((int) kSpatial[i]);
            base[i] = l.dim((int) inSpatial[i]);
        }
        long[] spatial =
                windowOutput(
                        base,
                        window,
                        op.windowStrides().orElse(null),
                        op.lhsDilation().orElse(null),
                        op.rhsDilation().orElse(null),
                        padding(op.padding().orElse(null), window.length));
        long[] out = filled(rank);
        for (int i = 0; i < spatial.length; i++) {
            out[(int) outSpatial[i]] = spatial[i];
        }
        long batch = l.dim((int) dn.inputBatchDimension());
        long groups = op.batchGroupCount();
        if (groups <= 0) {
            throw error("batch_group_count must be positive");
        }
        out[(int) dn.outputBatchDimension()] = batch == DYN ? DYN : batch / groups;
        out[(int) dn.outputFeatureDimension()] = r.dim((int) dn.kernelOutputFeatureDimension());
        return List.of(InferredType.shapeOnly(tensor(out, l.elementType())));
    }

    private static boolean isRealFloat(ElementType e) {
        return e.equals(FloatType.F16)
                || e.equals(FloatType.BF16)
                || e.equals(FloatType.F32)
                || e.equals(FloatType.F64);
    }

    private static List<InferredType> fft(FftOp op) {
        long[] length = op.fftLength();
        int fftRank = length.length;
        if (fftRank < 1 || fftRank > 3) {
            throw error("rank must be between 1 and 3, but got " + fftRank + ".");
        }
        TensorType t = tt(op.operand());
        FftType type = op.fftType();
        boolean rfft = type == FftType.RFFT;
        boolean irfft = type == FftType.IRFFT;
        ElementType e = t.elementType();
        if (rfft ? !isRealFloat(e) : !(e instanceof ComplexType)) {
            throw error(type + " does not take " + t.mlir());
        }
        ElementType re = rfft ? new ComplexType((FloatType) e) : irfft ? realOf(e) : e;
        if (t.rank() < fftRank) {
            throw error("operand rank must not be less than fft rank of " + fftRank);
        }
        long[] shape = t.dims();
        int last = shape.length - 1;
        int first = shape.length - fftRank;
        if (rfft) {
            for (int i = 0; i < fftRank; i++) {
                if (!Verifier.dimEq(shape[first + i], length[i])) {
                    throw error(
                            "RFFT requires innermost dimensions to be compatible with fft_length");
                }
            }
        }
        if (irfft) {
            for (int i = 0; i < fftRank - 1; i++) {
                if (!Verifier.dimEq(shape[first + i], length[i])) {
                    throw error(
                            "IRFFT requires non-final dimensions to be compatible with fft_length");
                }
            }
            long n = length[fftRank - 1];
            if ((!Verifier.dimEq(shape[last], 0) || n != 0)
                    && !Verifier.dimEq(shape[last], n / 2 + 1)) {
                throw error(
                        "IRFFT requires innermost dimension to be compatible with"
                                + " fft_length[-1]/2+1");
            }
        }
        if (rfft && length[fftRank - 1] != 0) {
            shape[last] = length[fftRank - 1] / 2 + 1;
        }
        if (irfft) {
            shape[last] = length[fftRank - 1];
        }
        long[] bounds = t.bounds();
        if (bounds != null && (rfft || irfft)) {
            bounds = bounds.clone();
            bounds[last] = DYN;
        }
        return exact(tensor(shape, re, bounds, t.encoding()));
    }

    private interface SliceDim {
        long at(long index);
    }

    /** {@code inferGatherReturnTypeComponents}. */
    private static List<InferredType> gather(
            Value operand, Value startIndices, GatherDimensionNumbers dn, SliceDim sliceDim) {
        TensorType o = tt(operand);
        TensorType s = tt(startIndices);
        long[] offsetDims = dn.offsetDims();
        long[] collapsed = dn.collapsedSliceDims();
        long[] batching = dn.operandBatchingDims();
        long ivd = dn.indexVectorDim();
        int sRank = s.rank() + (ivd == s.rank() ? 1 : 0);
        int resultRank = offsetDims.length + sRank - 1;
        for (long d : offsetDims) {
            if (d < 0 || d >= resultRank) {
                throw error(
                        "offset_dims "
                                + Arrays.toString(offsetDims)
                                + " out of the implied result rank "
                                + resultRank);
            }
        }
        long maxOperandDim = -1;
        for (long d : collapsed) {
            maxOperandDim = Math.max(maxOperandDim, d);
        }
        for (long d : batching) {
            maxOperandDim = Math.max(maxOperandDim, d);
        }
        List<Long> prefix = new ArrayList<>();
        for (int d = 0; d <= maxOperandDim; d++) {
            if (!contains(collapsed, d) && !contains(batching, d)) {
                prefix.add(sliceDim.at(d));
            }
        }
        long[] shape = new long[resultRank];
        int batchIndex = 0;
        for (int i = 0; i < resultRank; i++) {
            int offset = indexOf(offsetDims, i);
            if (offset >= 0) {
                shape[i] =
                        offset < prefix.size()
                                ? prefix.get(offset)
                                : sliceDim.at(offset + collapsed.length + batching.length);
            } else {
                int index = batchIndex++;
                if (index >= ivd) {
                    index++;
                }
                shape[i] = s.dim(index);
            }
        }
        long[] sb = s.bounds();
        long[] bounds = null;
        if (sb != null) {
            bounds = filled(resultRank);
            int sd = 0;
            for (int i = 0; i < resultRank; i++) {
                if (contains(offsetDims, i)) {
                    continue;
                }
                if (sd == ivd) {
                    sd++;
                }
                bounds[i] = sb[sd++];
            }
        }
        return exact(tensor(shape, o.elementType(), bounds, s.encoding()));
    }

    private static int indexOf(long[] values, long v) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] == v) {
                return i;
            }
        }
        return -1;
    }

    private static List<InferredType> setDimensionSize(SetDimensionSizeOp op) {
        TensorType t = tt(op.operand());
        if (tt(op.size()).rank() != 0) {
            throw error("size operand should be of rank-0");
        }
        long dim = op.dimension();
        checkDim(t, dim);
        int d = (int) dim;
        long[] shape = t.dims();
        long[] in = t.bounds();
        long[] bounds = in != null ? in.clone() : filled(t.rank());
        if (shape[d] != DYN) {
            bounds[d] = shape[d];
        }
        shape[d] = DYN;
        long[] size = constant(op.size());
        if (size != null && size.length == 1 && size[0] == bounds[d]) {
            shape[d] = size[0];
            bounds[d] = DYN;
        }
        return exact(tensor(shape, t.elementType(), bounds, null));
    }

    private static List<InferredType> rng(RngOp op) {
        TensorType b = tt(op.b());
        long[] shape = constant(op.shape());
        if (shape == null) {
            TensorType s = tt(op.shape());
            if (s.rank() != 1 || s.dim(0) == DYN) {
                throw error("shape operand must be a static 1-D tensor");
            }
            shape = filled((int) s.dim(0));
        }
        return exact(tensor(shape, b.elementType()));
    }

    /** The values of a constant integer operand, or null. */
    static long[] constant(Value v) {
        if (v.definingOp() != null
                && v.definingOp().asOp() instanceof ConstantOp c
                && c.value() instanceof DenseElementsAttr d
                && (d.type().elementType() instanceof IntegerType
                        || d.type().elementType() instanceof IndexType)) {
            long[] out = new long[(int) d.numElements()];
            for (int i = 0; i < out.length; i++) {
                out[i] = d.getLong(i);
            }
            return out;
        }
        return null;
    }
}

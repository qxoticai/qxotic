package com.qxotic.jota.tensor;

import com.qxotic.jota.*;
import com.qxotic.jota.impl.ViewTransforms;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.random.RandomKey;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

/**
 * Lazy tensor API.
 *
 * <p>Tensor operations build a graph and return new lazy tensors. Computation happens when {@link
 * #materialize()} is called. Methods that expose reduction axes use the wrap-around naming
 * convention: parameters named {@code _axis}/{@code _axes} are interpreted with wrap-around
 * semantics (negative indices allowed).
 */
public interface Tensor {

    DataType dataType();

    Layout layout();

    default Shape shape() {
        return layout().shape();
    }

    default Stride stride() {
        return layout().stride();
    }

    Device device();

    default long size() {
        return shape().size();
    }

    default boolean isScalar() {
        return shape().isScalar();
    }

    /**
     * Returns true if this tensor is a scalar broadcast (single value with all strides zero).
     *
     * <p>A scalar broadcast tensor stores a single element that is logically replicated across the
     * entire shape. This is memory-efficient for constants like zeros, ones, or fill values.
     */
    default boolean isScalarBroadcast() {
        Optional<MemoryView<?>> materialized = tryGetMaterialized();
        if (materialized.isPresent()) {
            MemoryView<?> view = materialized.get();
            return view.isBroadcasted()
                    && Arrays.stream(view.stride().toArray()).allMatch(s -> s == 0L);
        }
        return computation()
                .filter(ConstantComputation.class::isInstance)
                .map(
                        computation ->
                                Arrays.stream(layout().stride().toArray()).allMatch(s -> s == 0L))
                .orElse(false);
    }

    /** Returns raw bits for scalar constants without materializing, if available. */
    default java.util.OptionalLong scalarConstantBits() {
        java.util.Optional<ConstantComputation> constant =
                computation()
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast);
        if (constant.isPresent()) {
            return java.util.OptionalLong.of(constant.get().rawBits());
        }
        return java.util.OptionalLong.empty();
    }

    boolean isMaterialized();

    boolean isLazy();

    MemoryView<?> materialize();

    Optional<MemoryView<?>> tryGetMaterialized();

    Optional<LazyComputation> computation();

    default Tensor add(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.ADD,
                (left, right) -> Tracer.requireIROps().add(left, right),
                Tensor::add);
    }

    default Tensor add(int scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::add);
    }

    default Tensor add(long scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::add);
    }

    default Tensor add(float scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::add);
    }

    default Tensor add(double scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::add);
    }

    default Tensor subtract(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.SUBTRACT,
                (left, right) -> Tracer.requireIROps().subtract(left, right),
                Tensor::subtract);
    }

    default Tensor subtract(int scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::subtract);
    }

    default Tensor subtract(long scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::subtract);
    }

    default Tensor subtract(float scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::subtract);
    }

    default Tensor subtract(double scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::subtract);
    }

    default Tensor multiply(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.MULTIPLY,
                (left, right) -> Tracer.requireIROps().multiply(left, right),
                Tensor::multiply);
    }

    default Tensor multiply(int scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::multiply);
    }

    default Tensor multiply(long scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::multiply);
    }

    default Tensor multiply(float scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::multiply);
    }

    default Tensor multiply(double scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::multiply);
    }

    default Tensor divide(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.DIVIDE,
                (left, right) -> Tracer.requireIROps().divide(left, right),
                Tensor::divide);
    }

    default Tensor divide(int scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::divide);
    }

    default Tensor divide(long scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::divide);
    }

    default Tensor divide(float scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::divide);
    }

    default Tensor divide(double scalar) {
        return dispatchScalarBinaryOp(scalar, Tensor::divide);
    }

    default Tensor min(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.MIN,
                (left, right) -> Tracer.requireIROps().min(left, right),
                Tensor::min);
    }

    /**
     * Reduces all axes with minimum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor min() {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().min(this);
        }
        return Tracer.trace(this, Tensor::min);
    }

    /**
     * Reduces selected axes with minimum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor min(int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().min(this, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.min(_axis, _axes));
    }

    /**
     * Reduces selected axes with minimum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor min(boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().min(this, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.min(keepDims, _axis, _axes));
    }

    default Tensor max(Tensor other) {
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.MAX,
                (left, right) -> Tracer.requireIROps().max(left, right),
                Tensor::max);
    }

    /**
     * Reduces all axes with maximum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor max() {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().max(this);
        }
        return Tracer.trace(this, Tensor::max);
    }

    /**
     * Reduces selected axes with maximum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor max(int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().max(this, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.max(_axis, _axes));
    }

    /**
     * Reduces selected axes with maximum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor max(boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().max(this, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.max(keepDims, _axis, _axes));
    }

    /** Returns the global argmax index over the flattened tensor as scalar I64. */
    default Tensor argmax() {
        TensorTypeSemantics.requireNumericNonBool(dataType(), "argmax");
        if (shape().rank() == 0) {
            return Tensor.scalar(0L, DataType.I64);
        }
        Tensor flat = reshape(Shape.of(shape().size()));
        return flat.argmax(0, false);
    }

    /** Returns argmax indices along one axis as I64. */
    default Tensor argmax(int _axis) {
        return argmax(_axis, false);
    }

    /** Returns argmax indices along one axis as I64. */
    default Tensor argmax(int _axis, boolean keepDims) {
        TensorTypeSemantics.requireNumericNonBool(dataType(), "argmax");
        int axis = TensorSemantics.normalizeAxis(shape().rank(), _axis);
        long axisSize = shape().flatAt(axis);

        Tensor comparable = argReduceComparableView(this);
        Tensor extrema = comparable.max(true, axis);
        Tensor mask = comparable.equal(extrema.broadcast(shape()));
        Tensor axisIndices = axisIndexGrid(shape(), axis);
        Tensor sentinel = Tensor.full(axisSize, DataType.I64, shape());
        Tensor selected = mask.where(axisIndices, sentinel);
        return selected.min(keepDims, axis);
    }

    /** Returns the global argmin index over the flattened tensor as scalar I64. */
    default Tensor argmin() {
        TensorTypeSemantics.requireNumericNonBool(dataType(), "argmin");
        if (shape().rank() == 0) {
            return Tensor.scalar(0L, DataType.I64);
        }
        Tensor flat = reshape(Shape.of(shape().size()));
        return flat.argmin(0, false);
    }

    /** Returns argmin indices along one axis as I64. */
    default Tensor argmin(int _axis) {
        return argmin(_axis, false);
    }

    /** Returns argmin indices along one axis as I64. */
    default Tensor argmin(int _axis, boolean keepDims) {
        TensorTypeSemantics.requireNumericNonBool(dataType(), "argmin");
        int axis = TensorSemantics.normalizeAxis(shape().rank(), _axis);
        long axisSize = shape().flatAt(axis);

        Tensor comparable = argReduceComparableView(this);
        Tensor extrema = comparable.min(true, axis);
        Tensor mask = comparable.equal(extrema.broadcast(shape()));
        Tensor axisIndices = axisIndexGrid(shape(), axis);
        Tensor sentinel = Tensor.full(axisSize, DataType.I64, shape());
        Tensor selected = mask.where(axisIndices, sentinel);
        return selected.min(keepDims, axis);
    }

    default Tensor any() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        if (shape().rank() == 0) {
            return this;
        }
        Tensor reduced = sum(DataType.I64);
        return reduced.greaterThan(Tensor.scalar(0L, DataType.I64));
    }

    default Tensor all() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        if (shape().rank() == 0) {
            return this;
        }
        Tensor reduced = sum(DataType.I64);
        return reduced.equal(Tensor.scalar(shape().size(), DataType.I64));
    }

    default Tensor matmul(Tensor other) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().matmul(this, other);
        }
        return Tracer.trace(this, other, (a, b) -> a.matmul(b));
    }

    default Tensor batchedMatmul(Tensor other) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().batchedMatmul(this, other);
        }
        return Tracer.trace(this, other, (a, b) -> a.batchedMatmul(b));
    }

    /**
     * Dot product of two vectors with explicit accumulator type.
     *
     * <p>Strict semantics:
     *
     * <ul>
     *   <li>Both operands must be rank-1 (vectors)
     *   <li>Both vectors must have the same length and dtype
     *   <li>Input dtype must be numeric non-BOOL
     *   <li>Vectors must be non-empty
     * </ul>
     *
     * <p>Execution semantics: both inputs are cast to {@code accumulatorType}, multiplication is
     * performed in {@code accumulatorType}, and accumulation is also performed in {@code
     * accumulatorType}. The result is a scalar tensor of {@code accumulatorType}.
     *
     * <p>Example:
     *
     * <pre>{@code
     * Tensor a = Tensor.of(new int[] {50_000, 50_000});
     * Tensor b = Tensor.of(new int[] {50_000, 50_000});
     * Tensor out = a.dot(b, DataType.I64); // scalar I64, value 5_000_000_000
     * }</pre>
     */
    default Tensor dot(Tensor other, DataType accumulatorType) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(accumulatorType, "accumulatorType");

        TensorTypeSemantics.requireNumericNonBool(dataType(), "dot");
        TensorTypeSemantics.requireNumericNonBool(other.dataType(), "dot");
        if (dataType() != other.dataType()) {
            throw new IllegalArgumentException(
                    "dot requires same input dtypes, got "
                            + dataType()
                            + " and "
                            + other.dataType());
        }
        if (shape().rank() != 1 || other.shape().rank() != 1) {
            throw new IllegalArgumentException(
                    "dot requires rank-1 tensors, got " + shape() + " and " + other.shape());
        }
        long size = shape().flatAt(0);
        if (size != other.shape().flatAt(0)) {
            throw new IllegalArgumentException(
                    "dot requires equal vector lengths, got "
                            + size
                            + " and "
                            + other.shape().flatAt(0));
        }
        if (size == 0) {
            throw new IllegalArgumentException("dot requires non-empty vectors");
        }

        DataType accType =
                TensorTypeSemantics.resolveReductionAccumulator(dataType(), accumulatorType, "dot");
        Tensor left = cast(accType);
        Tensor right = other.cast(accType);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().dot(left, right, accType);
        }
        return Tracer.trace(left, right, (a, b) -> a.dot(b, accType));
    }

    /**
     * Dot product of two vectors using the default accumulator policy.
     *
     * <p>Default overload is intentionally floating-point only. For integral inputs, use {@link
     * #dot(Tensor, DataType)} to make the accumulator dtype explicit.
     */
    default Tensor dot(Tensor other) {
        Objects.requireNonNull(other, "other");
        if (!dataType().isFloatingPoint() || !other.dataType().isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "dot(other) is floating-point only; use dot(other, accumulatorType) for integral inputs");
        }
        return dot(other, dataType());
    }

    default Tensor to(Device device) {
        Objects.requireNonNull(device, "device");
        if (Tracer.isTracing()) {
            throw new UnsupportedOperationException(
                    "Tensor.to(Device) is a runtime transfer boundary and is not allowed inside tracing");
        }
        if (device.equals(device())) {
            return this;
        }
        return transferToDevice(this, device);
    }

    default Tensor contiguous() {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().contiguous(this);
        }
        return Tracer.trace(this, Tensor::contiguous);
    }

    /**
     * Returns a view of this tensor with a different shape.
     *
     * <p>This is a non-allocating reshape that returns a tensor sharing the same underlying memory.
     * The new shape must be compatible with the current layout (total size must match and the
     * layout must span a contiguous memory range).
     *
     * @param newShape the new shape for the view
     * @return a tensor with the new shape sharing the same memory
     * @throws IllegalArgumentException if a view cannot be created without copying
     */
    default Tensor view(Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(layout(), newShape);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    /**
     * Convenience overload for {@link #view(Shape)} with optional {@code -1} inference.
     *
     * <p>Rules:
     *
     * <ul>
     *   <li>At most one dimension may be {@code -1}
     *   <li>All other dimensions must be {@code >= 0}
     *   <li>The inferred size must divide total element count exactly
     *   <li>{@code view()} (no dims) is allowed only for one-element tensors and produces scalar
     *       shape {@code ()}
     * </ul>
     */
    default Tensor view(long... dims) {
        Objects.requireNonNull(dims, "dims");
        long total = shape().size();

        if (dims.length == 0) {
            if (total != 1) {
                throw new IllegalArgumentException(
                        "view() without dims requires exactly one element, got size=" + total);
            }
            return view(Shape.scalar());
        }

        return view(Shape.resolveShape(total, dims));
    }

    /**
     * Inserts a size-1 axis at {@code axis_}.
     *
     * <p>{@code axis_} uses output-shape wrap-around semantics (post-op indexing): for rank {@code
     * R}, valid values are in {@code [-(R+1), R]}.
     */
    default Tensor unsqueeze(int axis_) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.unsqueeze(layout(), axis_);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    /**
     * Removes a size-1 axis at {@code _axis}.
     *
     * <p>{@code _axis} uses input-shape wrap-around semantics. The selected axis/mode must have
     * size {@code 1}.
     */
    default Tensor squeeze(int _axis) {
        Shape currentShape = shape();
        int rank = currentShape.rank();
        if (rank == 0) {
            throw new IllegalArgumentException("cannot squeeze scalar tensor");
        }
        if (currentShape.size(_axis) != 1) {
            throw new IllegalArgumentException(
                    "cannot squeeze axis " + _axis + " with size " + currentShape.size(_axis));
        }
        return view(currentShape.remove(_axis));
    }

    /**
     * Removes all size-1 modes from the current (possibly nested) shape.
     *
     * <p>This is a view-only transform and preserves remaining nested structure.
     */
    default Tensor squeezeAll() {
        Shape currentShape = shape();
        if (currentShape.rank() == 0) {
            return this;
        }

        Shape squeezed = currentShape;
        for (int i = squeezed.rank() - 1; i >= 0; i--) {
            if (squeezed.size(i) == 1) {
                squeezed = squeezed.remove(i);
            }
        }
        return squeezed.equals(currentShape) ? this : view(squeezed);
    }

    default Tensor broadcast(Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.broadcast(layout(), targetShape);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    default Tensor expand(Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.expand(layout(), targetShape);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    default Tensor transpose(int _axis0, int _axis1) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.transpose(layout(), _axis0, _axis1);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    default Tensor permute(int... permutationIndices) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.permute(layout(), permutationIndices);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    default Tensor slice(int _axis, long start, long end) {
        return slice(_axis, start, end, 1);
    }

    default Tensor slice(int _axis, long start, long end, long indexStride) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.slice(layout(), dataType(), _axis, start, end, indexStride);
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().viewTransform(t, spec));
    }

    /**
     * Repeats this tensor along each axis.
     *
     * <p>Strict semantics: {@code repeats.length} must equal tensor rank and each repeat factor
     * must be {@code >= 1}. This method does not prepend dimensions or flatten implicitly. For
     * scalar tensors (rank 0), {@code repeat()} is the identity.
     */
    default Tensor repeat(long... repeats) {
        Objects.requireNonNull(repeats, "repeats");
        Shape currentShape = shape();
        Shape modeShape = currentShape.flattenModes();
        int rank = currentShape.rank();
        if (repeats.length != rank) {
            throw new IllegalArgumentException(
                    "repeat expects "
                            + rank
                            + " repeat factors, got "
                            + repeats.length
                            + " for shape "
                            + shape()
                            + " (mode shape "
                            + modeShape
                            + ")");
        }

        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = currentShape.size(i);
        }
        boolean identity = true;
        int expandedRank = 0;
        for (int i = 0; i < rank; i++) {
            if (repeats[i] < 1) {
                throw new IllegalArgumentException(
                        "repeat factors must be >= 1, got " + repeats[i] + " at axis " + i);
            }
            if (repeats[i] != 1) {
                identity = false;
                expandedRank++;
            }
            expandedRank++;
        }
        if (identity) {
            return this;
        }

        long[] unsqueezed = new long[expandedRank];
        long[] expanded = new long[expandedRank];
        long[] out = new long[rank];
        Tensor base = view(modeShape);

        int j = 0;
        for (int i = 0; i < rank; i++) {
            long dim = dims[i];
            long repeat = repeats[i];
            out[i] = Math.multiplyExact(dim, repeat);
            if (repeat == 1) {
                unsqueezed[j] = dim;
                expanded[j] = dim;
                j++;
                continue;
            }
            unsqueezed[j] = 1;
            expanded[j] = repeat;
            j++;
            unsqueezed[j] = dim;
            expanded[j] = dim;
            j++;
        }

        Tensor repeatedFlat =
                base.view(Shape.flat(unsqueezed))
                        .broadcast(Shape.flat(expanded))
                        .view(Shape.flat(out));
        return repeatedFlat.view(resolveRepeatedShape(currentShape, repeats, out));
    }

    private static Shape resolveRepeatedShape(Shape shape, long[] repeats, long[] outModeSizes) {
        if (shape.isFlat()) {
            return Shape.flat(outModeSizes);
        }
        Object[] modes = new Object[shape.rank()];
        for (int i = 0; i < modes.length; i++) {
            modes[i] = scaleMode(shape.modeAt(i), repeats[i]);
        }
        return Shape.of(modes);
    }

    private static Shape scaleMode(Shape mode, long repeat) {
        if (repeat == 1) {
            return mode;
        }
        long[] flatDims = mode.toArray();
        flatDims[0] = Math.multiplyExact(flatDims[0], repeat);
        if (mode.isFlat()) {
            return Shape.flat(flatDims);
        }
        return Shape.template(mode, flatDims);
    }

    /**
     * Repeats each element along one axis.
     *
     * <p>For example, {@code [a,b,c].repeatInterleave(2, 0)} produces {@code [a,a,b,b,c,c]}.
     */
    default Tensor repeatInterleave(long repeats, int _axis) {
        if (repeats < 1) {
            throw new IllegalArgumentException("repeatInterleave repeats must be >= 1");
        }
        if (repeats == 1) {
            return this;
        }

        Shape modeShape = shape().flattenModes();
        int rank = modeShape.rank();
        int axis = TensorSemantics.normalizeAxis(rank, _axis);
        long[] dims = modeShape.toArray();
        Tensor base = view(modeShape);

        long axisSize = dims[axis];
        long outAxisSize = Math.multiplyExact(axisSize, repeats);
        Tensor indexTensor =
                Tensor.iota(axisSize, DataType.I32)
                        .view(Shape.of(axisSize, 1))
                        .broadcast(Shape.of(axisSize, repeats))
                        .view(Shape.of(outAxisSize));

        return base.gather(indexTensor, axis);
    }

    /**
     * Concatenates tensors along an existing axis.
     *
     * <p>Strict semantics: all tensors must have the same dtype, device, rank, and equal sizes on
     * non-concatenated axes. No implicit broadcasting or rank padding is performed.
     */
    static Tensor concat(int _axis, Tensor first, Tensor second, Tensor... rest) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(rest, "rest");

        Shape firstShape = first.shape().flattenModes();
        int rank = firstShape.rank();
        int axis = TensorSemantics.normalizeAxis(rank, _axis);

        Tensor acc = asModeTensor(first, firstShape);
        Shape secondModeShape = second.shape().flattenModes();
        acc = concatPair(axis, acc, asModeTensor(second, secondModeShape));
        for (Tensor next : rest) {
            Objects.requireNonNull(next, "concat tensor");
            Shape nextModeShape = next.shape().flattenModes();
            acc = concatPair(axis, acc, asModeTensor(next, nextModeShape));
        }
        return acc;
    }

    /**
     * Stacks tensors along a new axis.
     *
     * <p>Strict semantics: all tensors must have the same dtype, device and shape. No implicit
     * broadcasting or rank padding is performed.
     */
    static Tensor stack(int _axis, Tensor first, Tensor second, Tensor... rest) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(rest, "rest");

        Shape firstShape = first.shape().flattenModes();
        int axis = TensorSemantics.normalizeAxis(firstShape.rank() + 1, _axis);
        Shape firstExpanded = insertAxisShape(firstShape, axis, 1L);

        Tensor[] expanded = new Tensor[2 + rest.length];
        expanded[0] = asModeTensor(first, firstShape).view(firstExpanded);
        Shape secondShape = second.shape().flattenModes();
        expanded[1] =
                asModeTensor(second, secondShape).view(insertAxisShape(secondShape, axis, 1L));
        for (int i = 0; i < rest.length; i++) {
            Tensor next = Objects.requireNonNull(rest[i], "stack tensor");
            Shape nextShape = next.shape().flattenModes();
            expanded[i + 2] =
                    asModeTensor(next, nextShape).view(insertAxisShape(nextShape, axis, 1L));
        }

        Tensor acc = concat(axis, expanded[0], expanded[1]);
        for (int i = 2; i < expanded.length; i++) {
            acc = concat(axis, acc, expanded[i]);
        }
        return acc;
    }

    /**
     * Splits a tensor into multiple views along one axis.
     *
     * <p>Strict semantics:
     *
     * <ul>
     *   <li>Split axis must refer to a flat (non-nested) mode
     *   <li>All explicit sizes must be {@code >= 1}
     *   <li>At most one size may be {@code -1} and is inferred from remaining size
     *   <li>Resolved sizes must sum exactly to axis size
     * </ul>
     */
    static Tensor[] split(
            int _axis, Tensor input, long firstSize, long secondSize, long... restSizes) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(restSizes, "restSizes");

        Shape shape = input.shape();
        int axis = TensorSemantics.normalizeAxis(shape.rank(), _axis);
        if (shape.modeAt(axis).rank() != 1) {
            throw new IllegalArgumentException(
                    "split axis cannot be nested: axis=" + _axis + ", mode=" + shape.modeAt(axis));
        }

        long[] sizes = new long[2 + restSizes.length];
        sizes[0] = firstSize;
        sizes[1] = secondSize;
        System.arraycopy(restSizes, 0, sizes, 2, restSizes.length);

        long axisSize = shape.size(axis);
        long[] resolved = resolveSplitSizes(axisSize, sizes);

        Tensor[] out = new Tensor[resolved.length];
        long start = 0;
        for (int i = 0; i < resolved.length; i++) {
            long size = resolved[i];
            out[i] = input.slice(axis, start, start + size);
            start += size;
        }
        return out;
    }

    private static Tensor asModeTensor(Tensor tensor, Shape modeShape) {
        return tensor.shape().equals(modeShape) ? tensor : tensor.view(modeShape);
    }

    private static Shape insertAxisShape(Shape base, int axis, long insertedSize) {
        long[] dims = base.toArray();
        long[] out = new long[dims.length + 1];
        for (int i = 0, j = 0; i < out.length; i++) {
            if (i == axis) {
                out[i] = insertedSize;
            } else {
                out[i] = dims[j++];
            }
        }
        return Shape.flat(out);
    }

    private static long[] resolveSplitSizes(long axisSize, long[] sizes) {
        int inferIndex = -1;
        long knownSum = 0;
        for (int i = 0; i < sizes.length; i++) {
            long s = sizes[i];
            if (s == -1) {
                if (inferIndex >= 0) {
                    throw new IllegalArgumentException("split allows at most one -1 size");
                }
                inferIndex = i;
                continue;
            }
            if (s < 1) {
                throw new IllegalArgumentException("split sizes must be >= 1 (or -1), got " + s);
            }
            knownSum = Math.addExact(knownSum, s);
        }

        long[] resolved = sizes.clone();
        if (inferIndex >= 0) {
            long inferred = axisSize - knownSum;
            if (inferred < 1) {
                throw new IllegalArgumentException(
                        "cannot infer split size: inferred size must be >= 1, got " + inferred);
            }
            resolved[inferIndex] = inferred;
            return resolved;
        }

        if (knownSum != axisSize) {
            throw new IllegalArgumentException(
                    "split sizes must sum to axis size " + axisSize + ", got " + knownSum);
        }
        return resolved;
    }

    private static Tensor concatPair(int axis, Tensor left, Tensor right) {
        if (left.dataType() != right.dataType()) {
            throw new IllegalArgumentException(
                    "concat requires matching dtypes, got "
                            + left.dataType()
                            + " and "
                            + right.dataType());
        }
        if (left.device() != right.device()) {
            throw new IllegalArgumentException(
                    "concat requires matching devices, got "
                            + left.device()
                            + " and "
                            + right.device());
        }

        Shape leftShape = left.shape();
        Shape rightShape = right.shape();
        if (leftShape.rank() != rightShape.rank()) {
            throw new IllegalArgumentException(
                    "concat requires equal ranks, got " + leftShape + " and " + rightShape);
        }

        long[] leftDims = leftShape.toArray();
        long[] rightDims = rightShape.toArray();
        for (int i = 0; i < leftDims.length; i++) {
            if (i == axis) {
                continue;
            }
            if (leftDims[i] != rightDims[i]) {
                throw new IllegalArgumentException(
                        "concat dimension mismatch at axis "
                                + i
                                + ": "
                                + leftDims[i]
                                + " vs "
                                + rightDims[i]);
            }
        }

        long leftAxis = leftDims[axis];
        long rightAxis = rightDims[axis];
        long outAxis = Math.addExact(leftAxis, rightAxis);

        long[] outDims = leftDims.clone();
        outDims[axis] = outAxis;
        Shape outShape = Shape.flat(outDims);

        Tensor idx = Tensor.iota(outAxis, DataType.I32);
        Tensor leftCount = Tensor.scalar(leftAxis, DataType.I32);
        Tensor zeroI32 = Tensor.scalar(0L, DataType.I32);
        Tensor leftMask = idx.lessThan(leftCount);

        Tensor leftIdx = leftMask.where(idx, zeroI32);
        Tensor rightIdx = leftMask.where(zeroI32, idx.subtract(leftCount));

        Tensor leftPart = left.gather(leftIdx, axis);
        Tensor rightPart = right.gather(rightIdx, axis);

        long[] maskDims = new long[leftDims.length];
        Arrays.fill(maskDims, 1L);
        maskDims[axis] = outAxis;
        Tensor axisCoord =
                Tensor.iota(outAxis, DataType.I32).view(Shape.flat(maskDims)).broadcast(outShape);
        Tensor mask = axisCoord.lessThan(leftCount);
        return mask.where(leftPart, rightPart);
    }

    /**
     * Returns a tensor with the specified shape.
     *
     * <p>This method tries to create a view first (non-allocating). If that's not possible because
     * the layout has gaps or doesn't span a contiguous memory range, it allocates new memory and
     * copies the data to create a contiguous tensor with the new shape.
     *
     * @param newShape the new shape (must have the same total number of elements)
     * @return a tensor with the new shape (may or may not share memory with the original)
     * @throws IllegalArgumentException if the total number of elements doesn't match
     */
    default Tensor reshape(Shape newShape) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().reshape(this, newShape);
        }
        return Tracer.trace(this, t -> Tracer.requireIROps().reshape(t, newShape));
    }

    default Tensor bitwiseNot() {
        TensorTypeSemantics.requireIntegral(dataType(), "bitwiseNot");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.BITWISE_NOT)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().bitwiseNot(this);
                            }
                            return Tracer.trace(this, Tensor::bitwiseNot);
                        });
    }

    default Tensor bitwiseAnd(Tensor other) {
        TensorTypeSemantics.requireSameIntegralType(dataType(), other.dataType(), "bitwiseAnd");
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.BITWISE_AND,
                (left, right) -> Tracer.requireIROps().bitwiseAnd(left, right),
                Tensor::bitwiseAnd);
    }

    default Tensor bitwiseOr(Tensor other) {
        TensorTypeSemantics.requireSameIntegralType(dataType(), other.dataType(), "bitwiseOr");
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.BITWISE_OR,
                (left, right) -> Tracer.requireIROps().bitwiseOr(left, right),
                Tensor::bitwiseOr);
    }

    default Tensor bitwiseXor(Tensor other) {
        TensorTypeSemantics.requireSameIntegralType(dataType(), other.dataType(), "bitwiseXor");
        return dispatchFoldedBinaryOp(
                other,
                BinaryOp.BITWISE_XOR,
                (left, right) -> Tracer.requireIROps().bitwiseXor(left, right),
                Tensor::bitwiseXor);
    }

    default Tensor leftShift(Tensor other) {
        return dispatchShiftBinaryOp(
                other,
                "leftShift",
                BinaryOp.LEFT_SHIFT,
                (left, right) -> Tracer.requireIROps().leftShift(left, right),
                Tensor::leftShift);
    }

    default Tensor rightShift(Tensor other) {
        return dispatchShiftBinaryOp(
                other,
                "rightShift",
                BinaryOp.RIGHT_SHIFT,
                (left, right) -> Tracer.requireIROps().rightShift(left, right),
                Tensor::rightShift);
    }

    default Tensor rightShiftUnsigned(Tensor other) {
        return dispatchShiftBinaryOp(
                other,
                "rightShiftUnsigned",
                BinaryOp.RIGHT_SHIFT_UNSIGNED,
                (left, right) -> Tracer.requireIROps().rightShiftUnsigned(left, right),
                Tensor::rightShiftUnsigned);
    }

    private Tensor normalizeShiftCountOperand(Tensor other, String opName) {
        TensorTypeSemantics.requireShiftOperandTypes(dataType(), other.dataType(), opName);
        if (other.dataType() != DataType.I32) {
            return other.cast(DataType.I32);
        }
        return other;
    }

    private Tensor dispatchBinaryOp(
            Tensor other,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        Tensor left = broadcastLeftScalar(this, other);
        Tensor right = broadcastRightScalar(this, other);
        if (Tracer.isTracing()) {
            return tracedOp.apply(left, right);
        }
        return Tracer.trace(left, right, eagerOp);
    }

    private Tensor dispatchScalarBinaryOp(
            Number scalar, BiFunction<Tensor, Tensor, Tensor> binaryOp) {
        Tensor scalarTensor;
        if (scalar instanceof Integer value) {
            scalarTensor = Tensor.broadcasted(value, shape());
        } else if (scalar instanceof Long value) {
            scalarTensor = Tensor.broadcasted(value, shape());
        } else if (scalar instanceof Float value) {
            scalarTensor = Tensor.broadcasted(value, shape());
        } else if (scalar instanceof Double value) {
            scalarTensor = Tensor.broadcasted(value, shape());
        } else {
            throw new IllegalArgumentException("Unsupported scalar type: " + scalar.getClass());
        }
        return binaryOp.apply(this, scalarTensor);
    }

    private Tensor dispatchFoldedBinaryOp(
            Tensor other,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        return ConstantFolder.tryFoldBinaryOp(this, other, foldOp)
                .orElseGet(() -> dispatchBinaryOp(other, tracedOp, eagerOp));
    }

    private Tensor dispatchFoldedCompareOp(
            Tensor other,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        return ConstantFolder.tryFoldCompareOp(this, other, foldOp)
                .orElseGet(() -> dispatchBinaryOp(other, tracedOp, eagerOp));
    }

    private Tensor dispatchShiftBinaryOp(
            Tensor other,
            String opName,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        Tensor normalizedOther = normalizeShiftCountOperand(other, opName);
        if (dataType() == normalizedOther.dataType()) {
            return dispatchFoldedBinaryOp(normalizedOther, foldOp, tracedOp, eagerOp);
        }
        return dispatchBinaryOp(normalizedOther, tracedOp, eagerOp);
    }

    default Tensor logicalNot() {
        TensorTypeSemantics.requireBool(dataType(), "logicalNot");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.LOGICAL_NOT)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().logicalNot(this);
                            }
                            return Tracer.trace(this, Tensor::logicalNot);
                        });
    }

    default Tensor logicalAnd(Tensor other) {
        TensorTypeSemantics.requireBooleanPair(dataType(), other.dataType(), "logicalAnd");
        return dispatchBinaryOp(
                other,
                (left, right) -> Tracer.requireIROps().logicalAnd(left, right),
                Tensor::logicalAnd);
    }

    default Tensor logicalOr(Tensor other) {
        TensorTypeSemantics.requireBooleanPair(dataType(), other.dataType(), "logicalOr");
        return dispatchBinaryOp(
                other,
                (left, right) -> Tracer.requireIROps().logicalOr(left, right),
                Tensor::logicalOr);
    }

    default Tensor logicalXor(Tensor other) {
        TensorTypeSemantics.requireBooleanPair(dataType(), other.dataType(), "logicalXor");
        return dispatchBinaryOp(
                other,
                (left, right) -> Tracer.requireIROps().logicalXor(left, right),
                Tensor::logicalXor);
    }

    default Tensor equal(Tensor other) {
        return dispatchFoldedCompareOp(
                other,
                BinaryOp.EQUAL,
                (left, right) -> Tracer.requireIROps().equal(left, right),
                Tensor::equal);
    }

    default Tensor lessThan(Tensor other) {
        return dispatchFoldedCompareOp(
                other,
                BinaryOp.LESS_THAN,
                (left, right) -> Tracer.requireIROps().lessThan(left, right),
                Tensor::lessThan);
    }

    default Tensor notEqual(Tensor other) {
        return equal(other).logicalNot();
    }

    default Tensor greaterThan(Tensor other) {
        return other.lessThan(this);
    }

    default Tensor lessThanOrEqual(Tensor other) {
        return other.lessThan(this).logicalNot();
    }

    default Tensor greaterThanOrEqual(Tensor other) {
        return lessThan(other).logicalNot();
    }

    /**
     * Elementwise conditional selection.
     *
     * <p>For each element, returns the corresponding value from {@code trueValue} when this tensor
     * (the condition) is true, otherwise from {@code falseValue}. The condition must be BOOL.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor where(Tensor trueValue, Tensor falseValue) {
        TensorTypeSemantics.requireBool(dataType(), "where condition");
        if (trueValue.dataType() != falseValue.dataType()) {
            throw new IllegalArgumentException(
                    "where requires true and false values to have the same type, got "
                            + trueValue.dataType()
                            + " and "
                            + falseValue.dataType());
        }
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().where(this, trueValue, falseValue);
        }
        return Tracer.trace(this, trueValue, falseValue, (c, t, f) -> c.where(t, f));
    }

    /**
     * Reduces all axes with sum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor sum(DataType accumulatorType) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().sum(this, accumulatorType);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType));
    }

    /**
     * Reduces selected axes with sum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor sum(DataType accumulatorType, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().sum(this, accumulatorType, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType, _axis, _axes));
    }

    /**
     * Reduces selected axes with sum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor sum(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().sum(this, accumulatorType, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType, keepDims, _axis, _axes));
    }

    /**
     * Reduces all axes with product.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor product(DataType accumulatorType) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().product(this, accumulatorType);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType));
    }

    /**
     * Reduces selected axes with product.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor product(DataType accumulatorType, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().product(this, accumulatorType, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType, _axis, _axes));
    }

    /**
     * Reduces selected axes with product.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    default Tensor product(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().product(this, accumulatorType, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType, keepDims, _axis, _axes));
    }

    /**
     * Reduces all axes with arithmetic mean.
     *
     * <p>Requires floating-point input. This is lazy and only executes on {@link #materialize()}.
     */
    default Tensor mean() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "mean");
        int rank = shape().rank();
        if (rank == 0) {
            return this;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return mean(false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    /**
     * Reduces selected axes with arithmetic mean.
     *
     * <p>Requires floating-point input. Parameters {@code _axis}/{@code _axes} use wrap-around axis
     * semantics.
     */
    default Tensor mean(int _axis, int... _axes) {
        return mean(false, _axis, _axes);
    }

    /**
     * Reduces selected axes with arithmetic mean.
     *
     * <p>Requires floating-point input. Parameters {@code _axis}/{@code _axes} use wrap-around axis
     * semantics.
     */
    default Tensor mean(boolean keepDims, int _axis, int... _axes) {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "mean");
        int[] axes = TensorSemantics.normalizeReductionAxes(shape().rank(), _axis, _axes);
        long count = 1L;
        for (int axis : axes) {
            count *= shape().flatAt(axis);
        }
        Tensor reduced = sum(dataType(), keepDims, _axis, _axes);
        return reduced.divide(Tensor.scalar((double) count, dataType()));
    }

    /**
     * Gathers elements from this tensor along the specified axis according to the indices.
     *
     * <p>This is commonly used for embedding lookups where: - this: [vocabSize, hiddenSize]
     * embedding table - indices: [batchSize, seqLen] token IDs - axis: 0 - returns: [batchSize,
     * seqLen, hiddenSize] embeddings
     *
     * @param indices the indices tensor (must be integral type)
     * @param _axis the axis along which to gather (wrap-around semantics)
     * @return the gathered tensor
     */
    default Tensor gather(Tensor indices, int _axis) {
        if (Tracer.isTracing()) {
            return Tracer.requireIROps().gather(this, indices, _axis);
        }
        return Tracer.trace(this, indices, (input, idx) -> input.gather(idx, _axis));
    }

    /**
     * Convenience method for embedding lookup (gather along axis 0).
     *
     * @param indices the indices tensor
     * @return the gathered embeddings
     */
    default Tensor embeddingLookup(Tensor indices) {
        return gather(indices, 0);
    }

    default Tensor cast(DataType targetType) {
        if (this.dataType() == targetType) {
            return this;
        }
        return ConstantFolder.tryFoldCast(this, targetType)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().cast(this, targetType);
                            }
                            return Tracer.trace(this, t -> t.cast(targetType));
                        });
    }

    default Tensor negate() {
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.NEGATE)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().negate(this);
                            }
                            return Tracer.trace(this, Tensor::negate);
                        });
    }

    default Tensor abs() {
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.ABS)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().abs(this);
                            }
                            return Tracer.trace(this, Tensor::abs);
                        });
    }

    default Tensor exp() {
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.EXP)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().exp(this);
                            }
                            return Tracer.trace(this, Tensor::exp);
                        });
    }

    default Tensor log() {
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.LOG)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().log(this);
                            }
                            return Tracer.trace(this, Tensor::log);
                        });
    }

    default Tensor sqrt() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "sqrt");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.SQRT)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().sqrt(this);
                            }
                            return Tracer.trace(this, Tensor::sqrt);
                        });
    }

    default Tensor square() {
        return multiply(this);
    }

    default Tensor sin() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "sin");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.SIN)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().sin(this);
                            }
                            return Tracer.trace(this, Tensor::sin);
                        });
    }

    default Tensor cos() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "cos");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.COS)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().cos(this);
                            }
                            return Tracer.trace(this, Tensor::cos);
                        });
    }

    default Tensor tanh() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "tanh");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.TANH)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().tanh(this);
                            }
                            return Tracer.trace(this, Tensor::tanh);
                        });
    }

    default Tensor relu() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "relu");
        return max(Tensor.full(0f, dataType(), shape()));
    }

    default Tensor sigmoid() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "sigmoid");
        return negate().exp().add(Tensor.scalar(1, dataType())).reciprocal();
    }

    default Tensor silu() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "silu");
        return multiply(sigmoid()); // x * sigmoid(x)
    }

    default Tensor gelu() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "gelu");
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        DataType dt = dataType();
        Tensor cubic = multiply(this).multiply(this);
        Tensor inner =
                cubic.multiply(Tensor.scalar(0.044715, dt))
                        .add(this)
                        .multiply(Tensor.scalar(0.7978845608, dt));
        return inner.tanh()
                .add(Tensor.scalar(1, dt))
                .multiply(this)
                .multiply(Tensor.scalar(0.5, dt));
    }

    /**
     * Broadcasts the left operand to match right's shape if left is a rank-0 scalar and right is
     * not.
     */
    private static Tensor broadcastLeftScalar(Tensor left, Tensor right) {
        if (!left.isScalar() || right.isScalar()) {
            return left;
        }
        // For constant computations, create a new broadcasted constant
        Optional<Tensor> constant =
                left.computation()
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast)
                        .map(c -> Tensor.full(c.value(), c.dataType(), right.shape()));
        if (constant.isPresent()) {
            return constant.get();
        }
        // For other scalars (e.g., IRTensor), use broadcast view transform
        return left.broadcast(right.shape());
    }

    /**
     * Broadcasts the right operand to match left's shape if right is a rank-0 scalar and left is
     * not.
     */
    private static Tensor broadcastRightScalar(Tensor left, Tensor right) {
        if (!right.isScalar() || left.isScalar()) {
            return right;
        }
        // For constant computations, create a new broadcasted constant
        Optional<Tensor> constant =
                right.computation()
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast)
                        .map(c -> Tensor.full(c.value(), c.dataType(), left.shape()));
        if (constant.isPresent()) {
            return constant.get();
        }
        // For other scalars (e.g., IRTensor), use broadcast view transform
        return right.broadcast(left.shape());
    }

    private static Tensor axisIndexGrid(Shape shape, int axis) {
        long[] indexShapeDims = new long[shape.rank()];
        Arrays.fill(indexShapeDims, 1L);
        indexShapeDims[axis] = shape.flatAt(axis);
        Tensor axisIndices =
                Tensor.iota(shape.flatAt(axis), DataType.I64).view(Shape.flat(indexShapeDims));
        return axisIndices.broadcast(shape);
    }

    private static Tensor argReduceComparableView(Tensor input) {
        DataType dataType = input.dataType();
        if (dataType == DataType.FP16 || dataType == DataType.BF16) {
            return input.cast(DataType.FP32);
        }
        return input;
    }

    default Tensor reciprocal() {
        TensorTypeSemantics.requireFloatingPoint(dataType(), "reciprocal");
        return ConstantFolder.tryFoldUnaryOp(this, UnaryOp.RECIPROCAL)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return Tracer.requireIROps().reciprocal(this);
                            }
                            return Tracer.trace(this, Tensor::reciprocal);
                        });
    }

    default Tensor reciprocal(DataType dataType) {
        if (!dataType.isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "reciprocal target type must be floating-point, got " + dataType);
        }
        return this.cast(dataType).reciprocal();
    }

    static Tensor of(MemoryView<?> view) {
        return new MaterializedTensor(view);
    }

    // ========== Tensor Creation Methods ==========

    /**
     * Creates a tensor filled with zeros.
     *
     * <p>Uses the default float type from {@link DataType#defaultFloat()}.
     *
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with zeros
     */
    static Tensor zeros(Shape shape) {
        return zeros(DataType.defaultFloat(), shape);
    }

    /**
     * Creates a tensor filled with zeros with the specified data type.
     *
     * @param dtype the data type
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with zeros
     */
    static Tensor zeros(DataType dtype, Shape shape) {
        return broadcasted(0, dtype, shape, Device.defaultDevice());
    }

    /**
     * Creates a tensor filled with ones.
     *
     * <p>Uses the default float type from {@link DataType#defaultFloat()}.
     *
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with ones
     */
    static Tensor ones(Shape shape) {
        return ones(DataType.defaultFloat(), shape);
    }

    /**
     * Creates a tensor filled with ones with the specified data type.
     *
     * @param dtype the data type
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with ones
     */
    static Tensor ones(DataType dtype, Shape shape) {
        return broadcasted(1, dtype, shape, Device.defaultDevice());
    }

    /**
     * Creates a 1D range tensor [0, 1, 2, ..., n-1] with I64 dtype.
     *
     * @param n number of elements (non-negative)
     * @return a lazy range tensor
     */
    static Tensor iota(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got: " + n);
        }
        Shape shape = Shape.flat(n);
        if (Tracer.isTracing()) {
            return new IRTensor(
                    new com.qxotic.jota.ir.tir.IotaConstant(n, DataType.I64, shape),
                    Device.defaultDevice());
        }
        Layout layout = Layout.rowMajor(shape);
        RangeComputation computation = new RangeComputation(n, Device.defaultDevice());
        return lazy(computation, DataType.I64, layout, Device.defaultDevice());
    }

    /**
     * Creates a 1D range tensor [0, 1, 2, ..., n-1] and casts to the target type.
     *
     * @param n number of elements (non-negative)
     * @param dataType target data type (integral or floating-point)
     * @return a lazy range tensor cast to the target type
     */
    static Tensor iota(long n, DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        if (dataType == DataType.BOOL || !(dataType.isIntegral() || dataType.isFloatingPoint())) {
            throw new IllegalArgumentException("Unsupported data type for iota: " + dataType);
        }
        if (Tracer.isTracing()) {
            return new IRTensor(
                    new com.qxotic.jota.ir.tir.IotaConstant(n, dataType, Shape.flat(n)),
                    Device.defaultDevice());
        }
        return iota(n).cast(dataType);
    }

    /**
     * Sets the thread-local RNG seed used by random methods that do not take an explicit {@link
     * RandomKey}.
     */
    static void manualSeed(long seed) {
        TensorRandomState.manualSeed(seed);
    }

    /** Alias of {@link #uniform(long, double, double, DataType)} with [0, 1) bounds. */
    static Tensor rand(long size, DataType dataType) {
        return rand(size, dataType, TensorRandomState.nextKey());
    }

    /** Alias of {@link #uniform(long, double, double, DataType, RandomKey)} with [0, 1) bounds. */
    static Tensor rand(long size, DataType dataType, RandomKey randomKey) {
        return randomUnitInterval(size, dataType, randomKey, "rand");
    }

    /** Alias of {@link #uniform(Shape, double, double, DataType)} with [0, 1) bounds. */
    static Tensor rand(Shape shape, DataType dataType) {
        return uniform(shape, 0.0, 1.0, dataType);
    }

    /** Alias of {@link #uniform(Shape, double, double, DataType, RandomKey)} with [0, 1) bounds. */
    static Tensor rand(Shape shape, DataType dataType, RandomKey randomKey) {
        return uniform(shape, 0.0, 1.0, dataType, randomKey);
    }

    /** Alias of {@link #normal(long, double, double, DataType)} with mean=0 and std=1. */
    static Tensor randn(long size, DataType dataType) {
        return randn(size, dataType, TensorRandomState.nextKey());
    }

    /**
     * Alias of {@link #normal(long, double, double, DataType, RandomKey)} with mean=0 and std=1.
     */
    static Tensor randn(long size, DataType dataType, RandomKey randomKey) {
        return randomStandardNormal(size, dataType, randomKey, "randn");
    }

    /** Alias of {@link #normal(Shape, double, double, DataType)} with mean=0 and std=1. */
    static Tensor randn(Shape shape, DataType dataType) {
        return normal(shape, 0.0, 1.0, dataType);
    }

    /**
     * Alias of {@link #normal(Shape, double, double, DataType, RandomKey)} with mean=0 and std=1.
     */
    static Tensor randn(Shape shape, DataType dataType, RandomKey randomKey) {
        return normal(shape, 0.0, 1.0, dataType, randomKey);
    }

    /** Alias of {@link #uniformInt(long, long, long, DataType)}. */
    static Tensor randInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return uniformInt(startInclusive, endExclusive, size, dataType);
    }

    /** Alias of {@link #uniformInt(long, long, long, DataType, RandomKey)}. */
    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        return uniformInt(startInclusive, endExclusive, size, dataType, randomKey);
    }

    /** Alias of {@link #uniformInt(long, long, Shape, DataType)}. */
    static Tensor randInt(long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        return uniformInt(startInclusive, endExclusive, shape, dataType);
    }

    /** Alias of {@link #uniformInt(long, long, Shape, DataType, RandomKey)}. */
    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        return uniformInt(startInclusive, endExclusive, shape, dataType, randomKey);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            long size, double startInclusive, double endExclusive, DataType dataType) {
        return uniform(size, startInclusive, endExclusive, dataType, TensorRandomState.nextKey());
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            long size,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        ensureValidRandomFloatRange(startInclusive, endExclusive, "uniform");
        Tensor unit = rand(size, dataType, randomKey);
        if (dataType == DataType.FP32) {
            float span = (float) (endExclusive - startInclusive);
            float start = (float) startInclusive;
            return unit.multiply(span).add(start);
        }
        double span = endExclusive - startInclusive;
        return unit.multiply(span).add(startInclusive);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            Shape shape, double startInclusive, double endExclusive, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return uniform(shape, startInclusive, endExclusive, dataType, TensorRandomState.nextKey());
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            Shape shape,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return uniform(shape.size(), startInclusive, endExclusive, dataType, randomKey).view(shape);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(long size, double mean, double std, DataType dataType) {
        return normal(size, mean, std, dataType, TensorRandomState.nextKey());
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        ensureValidNormalParams(mean, std, "normal");
        Tensor standard = randn(size, dataType, randomKey);
        if (dataType == DataType.FP32) {
            return standard.multiply((float) std).add((float) mean);
        }
        return standard.multiply(std).add(mean);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(Shape shape, double mean, double std, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return normal(shape, mean, std, dataType, TensorRandomState.nextKey());
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return normal(shape.size(), mean, std, dataType, randomKey).view(shape);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return uniformInt(
                startInclusive, endExclusive, size, dataType, TensorRandomState.nextKey());
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "uniformInt");
        ensureValidRandIntRange(startInclusive, endExclusive, dataType, "uniformInt");

        long range;
        try {
            range = Math.subtractExact(endExclusive, startInclusive);
        } catch (ArithmeticException ex) {
            throw new IllegalArgumentException(
                    "uniformInt range overflow: [" + startInclusive + ", " + endExclusive + ")",
                    ex);
        }

        Tensor uniform = randomUnitInterval(size, DataType.FP64, randomKey, "uniformInt");
        Tensor offset = uniform.multiply((double) range).cast(DataType.I64);
        Tensor shifted = offset.add(startInclusive);
        return dataType == DataType.I64 ? shifted : shifted.cast(dataType);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return uniformInt(
                        startInclusive,
                        endExclusive,
                        shape.size(),
                        dataType,
                        TensorRandomState.nextKey())
                .view(shape);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return uniformInt(startInclusive, endExclusive, shape.size(), dataType, randomKey)
                .view(shape);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(long size, double mean, double std, DataType dataType) {
        return normalInt(size, mean, std, dataType, TensorRandomState.nextKey());
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "normalInt");
        ensureValidNormalParams(mean, std, "normalInt");

        Tensor fp = normal(size, mean, std, DataType.FP64, randomKey);
        Tensor rounded =
                fp.greaterThanOrEqual(Tensor.scalar(0.0, DataType.FP64))
                        .where(fp.add(0.5), fp.subtract(0.5))
                        .cast(DataType.I64);
        Tensor clamped =
                rounded.max(Tensor.scalar(randomIntMin(dataType), DataType.I64))
                        .min(Tensor.scalar(randomIntMax(dataType), DataType.I64));
        return dataType == DataType.I64 ? clamped : clamped.cast(dataType);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(Shape shape, double mean, double std, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return normalInt(shape, mean, std, dataType, TensorRandomState.nextKey());
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return normalInt(shape.size(), mean, std, dataType, randomKey).view(shape);
    }

    /**
     * Creates a tensor filled with the specified float value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (FP32)
     */
    static Tensor full(float value, Shape shape) {
        return broadcasted(Float.valueOf(value), DataType.FP32, shape, Device.defaultDevice());
    }

    /**
     * Creates a tensor filled with the specified double value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (FP64)
     */
    static Tensor full(double value, Shape shape) {
        return broadcasted(Double.valueOf(value), DataType.FP64, shape, Device.defaultDevice());
    }

    /**
     * Creates a tensor filled with the specified long value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (I64)
     */
    static Tensor full(long value, Shape shape) {
        return broadcasted(Long.valueOf(value), DataType.I64, shape, Device.defaultDevice());
    }

    /**
     * Creates a tensor filled with the specified int value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (I32)
     */
    static Tensor full(int value, Shape shape) {
        return broadcasted(Integer.valueOf(value), DataType.I32, shape, Device.defaultDevice());
    }

    /**
     * Creates a tensor filled with the specified value and data type.
     *
     * @param value the fill value
     * @param dtype the data type
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value
     */
    static Tensor full(Number value, DataType dtype, Shape shape) {
        return broadcasted(value, dtype, shape, Device.defaultDevice());
    }

    // ========== Broadcasted and Scalar ==========

    static Tensor broadcasted(int value, Shape shape) {
        return broadcasted(Integer.valueOf(value), DataType.I32, shape, Device.defaultDevice());
    }

    static Tensor broadcasted(long value, Shape shape) {
        return broadcasted(Long.valueOf(value), DataType.I64, shape, Device.defaultDevice());
    }

    static Tensor broadcasted(float value, Shape shape) {
        return broadcasted(Float.valueOf(value), DataType.FP32, shape, Device.defaultDevice());
    }

    static Tensor broadcasted(double value, Shape shape) {
        return broadcasted(Double.valueOf(value), DataType.FP64, shape, Device.defaultDevice());
    }

    static Tensor scalar(int value) {
        return broadcasted(
                Integer.valueOf(value), DataType.I32, Shape.scalar(), Device.defaultDevice());
    }

    static Tensor scalar(float value) {
        return broadcasted(
                Float.valueOf(value), DataType.FP32, Shape.scalar(), Device.defaultDevice());
    }

    static Tensor scalar(double value) {
        return broadcasted(
                Double.valueOf(value), DataType.FP64, Shape.scalar(), Device.defaultDevice());
    }

    static Tensor scalar(long value) {
        return broadcasted(
                Long.valueOf(value), DataType.I64, Shape.scalar(), Device.defaultDevice());
    }

    /**
     * Creates a scalar tensor with the specified data type.
     *
     * <p>The primitive value is used as a carrier and cast to the target type.
     *
     * @param value the scalar value (used as carrier)
     * @param dtype the target data type
     * @return a scalar tensor with the specified type
     */
    static Tensor scalar(double value, DataType dtype) {
        return broadcasted(Double.valueOf(value), dtype, Shape.scalar(), Device.defaultDevice());
    }

    /**
     * Creates a scalar tensor with the specified data type.
     *
     * <p>The primitive value is used as a carrier and cast to the target type.
     *
     * @param value the scalar value (used as carrier)
     * @param dtype the target data type
     * @return a scalar tensor with the specified type
     */
    static Tensor scalar(long value, DataType dtype) {
        return broadcasted(Long.valueOf(value), dtype, Shape.scalar(), Device.defaultDevice());
    }

    static Tensor lazy(LazyComputation computation, DataType dtype, Layout layout, Device device) {
        return new LazyTensor(computation, dtype, layout, device);
    }

    private static Tensor broadcasted(Number value, DataType dataType, Shape shape, Device device) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(device, "device");
        Layout layout = Layout.of(shape, Stride.zeros(shape));

        if (Tracer.isTracing()) {
            return createIRScalarConstant(value, dataType, shape, device);
        }

        ConstantComputation computation = ConstantComputation.of(value, dataType, shape, device);
        return lazy(computation, dataType, layout, device);
    }

    private static void ensureSupportedRandomFloat(DataType dtype, String opName) {
        if (dtype != DataType.FP32 && dtype != DataType.FP64) {
            throw new IllegalArgumentException(opName + " supports FP32/FP64 only, got: " + dtype);
        }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static Tensor transferToDevice(Tensor source, Device targetDevice) {
        MemoryView srcView = source.materialize();
        Environment environment = Environment.current();
        MemoryDomain srcDomain = environment.runtimeFor(source.device()).memoryDomain();
        MemoryDomain dstDomain = environment.runtimeFor(targetDevice).memoryDomain();

        MemoryView dstView =
                MemoryView.of(
                        dstDomain
                                .memoryAllocator()
                                .allocateMemory(source.dataType(), source.shape()),
                        source.dataType(),
                        Layout.rowMajor(source.shape()));

        MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
        return Tensor.of(dstView);
    }

    private static Tensor randomUnitInterval(
            long size, DataType dataType, RandomKey randomKey, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);
        Shape shape = shapeFromSize(size, opName);
        Layout layout = Layout.rowMajor(shape);
        RandomComputation computation =
                new RandomComputation(shape, dataType, Device.defaultDevice(), randomKey);
        return lazy(computation, dataType, layout, Device.defaultDevice());
    }

    private static Tensor randomStandardNormal(
            long size, DataType dataType, RandomKey randomKey, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);

        RandomKey k0 = randomKey.split(0L);
        RandomKey k1 = randomKey.split(1L);
        Tensor u1 = randomUnitInterval(size, dataType, k0, opName);
        Tensor u2 = randomUnitInterval(size, dataType, k1, opName);

        if (dataType == DataType.FP64) {
            Tensor r = u2.multiply(-1.0).add(1.0).log().multiply(-2.0).sqrt();
            Tensor theta = u1.multiply(2.0 * Math.PI);
            return theta.cos().multiply(r);
        }

        Tensor r = u2.multiply(-1.0f).add(1.0f).log().multiply(-2.0f).sqrt();
        Tensor theta = u1.multiply((float) (2.0 * Math.PI));
        return theta.cos().multiply(r);
    }

    private static void ensureSupportedRandomInt(DataType dtype, String opName) {
        if (dtype == DataType.BOOL || !dtype.isIntegral()) {
            throw new IllegalArgumentException(
                    opName + " supports I8/I16/I32/I64 only, got: " + dtype);
        }
    }

    private static void ensureValidRandomFloatRange(
            double startInclusive, double endExclusive, String opName) {
        if (!Double.isFinite(startInclusive) || !Double.isFinite(endExclusive)) {
            throw new IllegalArgumentException(opName + " requires finite bounds");
        }
        if (!(endExclusive > startInclusive)) {
            throw new IllegalArgumentException(
                    opName
                            + " requires endExclusive > startInclusive, got ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ")");
        }
    }

    private static void ensureValidNormalParams(double mean, double std, String opName) {
        if (!Double.isFinite(mean) || !Double.isFinite(std)) {
            throw new IllegalArgumentException(opName + " requires finite mean/std");
        }
        if (!(std > 0.0)) {
            throw new IllegalArgumentException(opName + " requires std > 0, got: " + std);
        }
    }

    private static void ensureValidRandIntRange(
            long startInclusive, long endExclusive, DataType dataType, String opName) {
        if (endExclusive <= startInclusive) {
            throw new IllegalArgumentException(
                    opName
                            + " requires endExclusive > startInclusive, got ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ")");
        }
        long min = randomIntMin(dataType);
        long max = randomIntMax(dataType);
        long upperInclusive = endExclusive - 1L;
        if (startInclusive < min || upperInclusive > max) {
            throw new IllegalArgumentException(
                    opName
                            + " range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") does not fit dtype "
                            + dataType);
        }
    }

    private static Shape shapeFromSize(long size, String opName) {
        if (size < 0) {
            throw new IllegalArgumentException(
                    opName + " requires non-negative size, got: " + size);
        }
        return Shape.flat(size);
    }

    private static long randomIntMin(DataType dataType) {
        if (dataType == DataType.I8) {
            return Byte.MIN_VALUE;
        }
        if (dataType == DataType.I16) {
            return Short.MIN_VALUE;
        }
        if (dataType == DataType.I32) {
            return Integer.MIN_VALUE;
        }
        if (dataType == DataType.I64) {
            return Long.MIN_VALUE;
        }
        throw new IllegalArgumentException("Unsupported random integer dtype: " + dataType);
    }

    private static long randomIntMax(DataType dataType) {
        if (dataType == DataType.I8) {
            return Byte.MAX_VALUE;
        }
        if (dataType == DataType.I16) {
            return Short.MAX_VALUE;
        }
        if (dataType == DataType.I32) {
            return Integer.MAX_VALUE;
        }
        if (dataType == DataType.I64) {
            return Long.MAX_VALUE;
        }
        throw new IllegalArgumentException("Unsupported random integer dtype: " + dataType);
    }

    private static Tensor createIRScalarConstant(
            Number value, DataType dataType, Shape shape, Device device) {
        com.qxotic.jota.ir.tir.ScalarConstant scalar;
        long rawBits;
        if (dataType == DataType.FP32) {
            rawBits = Float.floatToIntBits(value.floatValue());
            scalar = com.qxotic.jota.ir.tir.ScalarConstant.broadcast(rawBits, dataType, shape);
        } else if (dataType == DataType.FP64) {
            rawBits = Double.doubleToLongBits(value.doubleValue());
            scalar = com.qxotic.jota.ir.tir.ScalarConstant.broadcast(rawBits, dataType, shape);
        } else if (dataType == DataType.BOOL
                || dataType == DataType.I8
                || dataType == DataType.I16
                || dataType == DataType.I32) {
            rawBits = value.longValue();
            scalar = com.qxotic.jota.ir.tir.ScalarConstant.broadcast(rawBits, dataType, shape);
        } else if (dataType == DataType.I64) {
            rawBits = value.longValue();
            scalar = com.qxotic.jota.ir.tir.ScalarConstant.broadcast(rawBits, dataType, shape);
        } else if (dataType == DataType.FP16 || dataType == DataType.BF16) {
            rawBits = (long) Float.floatToIntBits(value.floatValue());
            scalar = com.qxotic.jota.ir.tir.ScalarConstant.broadcast(rawBits, dataType, shape);
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
        return new IRTensor(scalar, device);
    }

    // ========== Array Creation (Eager) ==========

    /**
     * Creates a tensor from a float array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and FP32 dtype
     */
    static Tensor of(float[] data) {
        return of(data, Shape.flat(data.length));
    }

    /**
     * Creates a tensor from a float array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with FP32 dtype
     */
    static Tensor of(float[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + data.length + " does not match shape size " + shape.size());
        }
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyFloatArray(memoryDomain, data, shape);
        return of(view);
    }

    /**
     * Creates a tensor from a double array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and FP64 dtype
     */
    static Tensor of(double[] data) {
        return of(data, Shape.flat(data.length));
    }

    /**
     * Creates a tensor from a double array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with FP64 dtype
     */
    static Tensor of(double[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + data.length + " does not match shape size " + shape.size());
        }
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyDoubleArray(memoryDomain, data, shape);
        return of(view);
    }

    /**
     * Creates a tensor from an int array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and I32 dtype
     */
    static Tensor of(int[] data) {
        return of(data, Shape.flat(data.length));
    }

    /**
     * Creates a tensor from an int array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with I32 dtype
     */
    static Tensor of(int[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + data.length + " does not match shape size " + shape.size());
        }
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyIntArray(memoryDomain, data, shape);
        return of(view);
    }

    /**
     * Creates a tensor from a long array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and I64 dtype
     */
    static Tensor of(long[] data) {
        return of(data, Shape.flat(data.length));
    }

    /**
     * Creates a tensor from a long array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with I64 dtype
     */
    static Tensor of(long[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + data.length + " does not match shape size " + shape.size());
        }
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyLongArray(memoryDomain, data, shape);
        return of(view);
    }

    /**
     * Creates a tensor from a boolean array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and BOOL dtype
     */
    static Tensor of(boolean[] data) {
        return of(data, Shape.flat(data.length));
    }

    /**
     * Creates a tensor from a boolean array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with BOOL dtype
     */
    static Tensor of(boolean[] data, Shape shape) {
        if (data.length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + data.length + " does not match shape size " + shape.size());
        }
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyBooleanArray(memoryDomain, data, shape);
        return of(view);
    }

    private static <B> MemoryView<B> copyFloatArray(
            MemoryDomain<B> memoryDomain, float[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.FP32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Float.BYTES);
        return MemoryView.of(dst, DataType.FP32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyDoubleArray(
            MemoryDomain<B> memoryDomain, double[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.FP64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Double.BYTES);
        return MemoryView.of(dst, DataType.FP64, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyIntArray(
            MemoryDomain<B> memoryDomain, int[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.I32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Integer.BYTES);
        return MemoryView.of(dst, DataType.I32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyLongArray(
            MemoryDomain<B> memoryDomain, long[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.I64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Long.BYTES);
        return MemoryView.of(dst, DataType.I64, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyBooleanArray(
            MemoryDomain<B> memoryDomain, boolean[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.BOOL, data.length);
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = data[i] ? (byte) 1 : 0;
        }
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(bytes));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Byte.BYTES);
        return MemoryView.of(dst, DataType.BOOL, Layout.rowMajor(shape));
    }

    @Deprecated(forRemoval = true, since = "0.1")
    default Tensor realize() {
        materialize();
        return this;
    }
}

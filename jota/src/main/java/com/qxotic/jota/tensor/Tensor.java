package com.qxotic.jota.tensor;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.random.RandomKey;

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

    MemoryView<?> materialize();

    // region Instance Operations
    // region Binary Operations

    Tensor add(Tensor other);

    Tensor add(int scalar);

    Tensor add(long scalar);

    Tensor add(float scalar);

    Tensor add(double scalar);

    Tensor subtract(Tensor other);

    Tensor subtract(int scalar);

    Tensor subtract(long scalar);

    Tensor subtract(float scalar);

    Tensor subtract(double scalar);

    Tensor multiply(Tensor other);

    Tensor multiply(int scalar);

    Tensor multiply(long scalar);

    Tensor multiply(float scalar);

    Tensor multiply(double scalar);

    Tensor divide(Tensor other);

    Tensor divide(int scalar);

    Tensor divide(long scalar);

    Tensor divide(float scalar);

    Tensor divide(double scalar);

    Tensor min(Tensor other);

    /**
     * Reduces all axes with minimum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    Tensor min();

    /**
     * Reduces selected axes with minimum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor min(int _axis, int... _axes);

    /**
     * Reduces selected axes with minimum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor min(boolean keepDims, int _axis, int... _axes);

    Tensor max(Tensor other);

    /**
     * Reduces all axes with maximum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    Tensor max();

    /**
     * Reduces selected axes with maximum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor max(int _axis, int... _axes);

    /**
     * Reduces selected axes with maximum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor max(boolean keepDims, int _axis, int... _axes);

    // endregion Binary Operations
    // region Reduction / Linear Algebra Operations

    /** Returns the global argmax index over the flattened tensor as scalar I64. */
    Tensor argmax();

    /** Returns argmax indices along one axis as I64. */
    Tensor argmax(int _axis);

    /** Returns argmax indices along one axis as I64. */
    Tensor argmax(int _axis, boolean keepDims);

    /** Returns the global argmin index over the flattened tensor as scalar I64. */
    Tensor argmin();

    /** Returns argmin indices along one axis as I64. */
    Tensor argmin(int _axis);

    /** Returns argmin indices along one axis as I64. */
    Tensor argmin(int _axis, boolean keepDims);

    Tensor any();

    Tensor all();

    Tensor matmul(Tensor other);

    Tensor batchedMatmul(Tensor other);

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
    Tensor dot(Tensor other, DataType accumulatorType);

    /**
     * Dot product of two vectors using the default accumulator policy.
     *
     * <p>Default overload is intentionally floating-point only. For integral inputs, use {@link
     * #dot(Tensor, DataType)} to make the accumulator dtype explicit.
     */
    Tensor dot(Tensor other);

    Tensor to(Device device);

    Tensor contiguous();

    // endregion Reduction / Linear Algebra Operations
    // region Movement / Shape Operations

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
    Tensor view(Shape newShape);

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
    Tensor view(long... dims);

    /**
     * Inserts a size-1 axis at {@code axis_}.
     *
     * <p>{@code axis_} uses output-shape wrap-around semantics (post-op indexing): for rank {@code
     * R}, valid values are in {@code [-(R+1), R]}.
     */
    Tensor unsqueeze(int axis_);

    /**
     * Removes a size-1 axis at {@code _axis}.
     *
     * <p>{@code _axis} uses input-shape wrap-around semantics. The selected axis/mode must have
     * size {@code 1}.
     */
    Tensor squeeze(int _axis);

    /**
     * Removes all size-1 modes from the current (possibly nested) shape.
     *
     * <p>This is a view-only transform and preserves remaining nested structure.
     */
    Tensor squeezeAll();

    Tensor broadcast(Shape targetShape);

    Tensor expand(Shape targetShape);

    Tensor transpose(int _axis0, int _axis1);

    Tensor permute(int... permutationIndices);

    Tensor slice(int _axis, long start, long end);

    Tensor slice(int _axis, long start, long end, long indexStride);

    /**
     * Repeats this tensor along each axis.
     *
     * <p>Strict semantics: {@code repeats.length} must equal tensor rank and each repeat factor
     * must be {@code >= 1}. This method does not prepend dimensions or flatten implicitly. For
     * scalar tensors (rank 0), {@code repeat()} is the identity.
     */
    Tensor repeat(long... repeats);

    /**
     * Repeats each element along one axis.
     *
     * <p>For example, {@code [a,b,c].repeatInterleave(2, 0)} produces {@code [a,a,b,b,c,c]}.
     */
    Tensor repeatInterleave(long repeats, int _axis);

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
    Tensor reshape(Shape newShape);

    // endregion Movement / Shape Operations
    // region Static Structural Operations

    /**
     * Concatenates tensors along an existing axis.
     *
     * <p>Strict semantics: all tensors must have the same dtype, device, rank, and equal sizes on
     * non-concatenated axes. No implicit broadcasting or rank padding is performed.
     */
    static Tensor concat(int _axis, Tensor first, Tensor second, Tensor... rest) {
        return TensorFactory.concat(_axis, first, second, rest);
    }

    /**
     * Stacks tensors along a new axis.
     *
     * <p>Strict semantics: all tensors must have the same dtype, device and shape. No implicit
     * broadcasting or rank padding is performed.
     */
    static Tensor stack(int _axis, Tensor first, Tensor second, Tensor... rest) {
        return TensorFactory.stack(_axis, first, second, rest);
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
        return TensorFactory.split(_axis, input, firstSize, secondSize, restSizes);
    }

    // endregion Static Structural Operations
    // region Bitwise / Logical / Comparison Operations

    Tensor bitwiseNot();

    Tensor bitwiseAnd(Tensor other);

    Tensor bitwiseOr(Tensor other);

    Tensor bitwiseXor(Tensor other);

    Tensor leftShift(Tensor other);

    Tensor rightShift(Tensor other);

    Tensor rightShiftUnsigned(Tensor other);

    Tensor logicalNot();

    Tensor logicalAnd(Tensor other);

    Tensor logicalOr(Tensor other);

    Tensor logicalXor(Tensor other);

    Tensor equal(Tensor other);

    Tensor lessThan(Tensor other);

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

    // endregion Bitwise / Logical / Comparison Operations
    // region Selection / Reduction Operations

    /**
     * Elementwise conditional selection.
     *
     * <p>For each element, returns the corresponding value from {@code trueValue} when this tensor
     * (the condition) is true, otherwise from {@code falseValue}. The condition must be BOOL.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    Tensor where(Tensor trueValue, Tensor falseValue);

    /**
     * Reduces all axes with sum.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    Tensor sum(DataType accumulatorType);

    /**
     * Reduces selected axes with sum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor sum(DataType accumulatorType, int _axis, int... _axes);

    /**
     * Reduces selected axes with sum.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor sum(DataType accumulatorType, boolean keepDims, int _axis, int... _axes);

    /**
     * Reduces all axes with product.
     *
     * <p>This is lazy and only executes on {@link #materialize()}.
     */
    Tensor product(DataType accumulatorType);

    /**
     * Reduces selected axes with product.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor product(DataType accumulatorType, int _axis, int... _axes);

    /**
     * Reduces selected axes with product.
     *
     * <p>Parameters {@code _axis}/{@code _axes} use wrap-around axis semantics.
     */
    Tensor product(DataType accumulatorType, boolean keepDims, int _axis, int... _axes);

    /**
     * Reduces all axes with arithmetic mean.
     *
     * <p>Requires floating-point input. This is lazy and only executes on {@link #materialize()}.
     */
    Tensor mean();

    /**
     * Reduces selected axes with arithmetic mean.
     *
     * <p>Requires floating-point input. Parameters {@code _axis}/{@code _axes} use wrap-around axis
     * semantics.
     */
    Tensor mean(int _axis, int... _axes);

    /**
     * Reduces selected axes with arithmetic mean.
     *
     * <p>Requires floating-point input. Parameters {@code _axis}/{@code _axes} use wrap-around axis
     * semantics.
     */
    Tensor mean(boolean keepDims, int _axis, int... _axes);

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
    Tensor gather(Tensor indices, int _axis);

    /**
     * Convenience method for embedding lookup (gather along axis 0).
     *
     * @param indices the indices tensor
     * @return the gathered embeddings
     */
    Tensor embeddingLookup(Tensor indices);

    // endregion Selection / Reduction Operations
    // region Elementwise Unary Operations

    Tensor cast(DataType targetType);

    Tensor negate();

    Tensor abs();

    Tensor exp();

    Tensor log();

    Tensor sqrt();

    default Tensor square() {
        return multiply(this);
    }

    Tensor sin();

    Tensor cos();

    Tensor tanh();

    Tensor relu();

    Tensor sigmoid();

    Tensor silu();

    Tensor gelu();

    Tensor reciprocal();

    Tensor reciprocal(DataType dataType);

    // endregion Elementwise Unary Operations
    // endregion Instance Operations

    // region Static Creation Methods
    // region Core Creation

    static Tensor of(MemoryView<?> view) {
        return TensorFactory.of(view);
    }

    // endregion Core Creation
    // region Constant / Fill Creation

    /**
     * Creates a tensor filled with zeros.
     *
     * <p>Uses the default float type from {@link DataType#defaultFloat()}.
     *
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with zeros
     */
    static Tensor zeros(Shape shape) {
        return TensorFactory.zeros(shape);
    }

    /**
     * Creates a tensor filled with zeros with the specified data type.
     *
     * @param dtype the data type
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with zeros
     */
    static Tensor zeros(DataType dtype, Shape shape) {
        return TensorFactory.zeros(dtype, shape);
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
        return TensorFactory.ones(shape);
    }

    /**
     * Creates a tensor filled with ones with the specified data type.
     *
     * @param dtype the data type
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with ones
     */
    static Tensor ones(DataType dtype, Shape shape) {
        return TensorFactory.ones(dtype, shape);
    }

    /**
     * Creates a 1D range tensor [0, 1, 2, ..., n-1] with I64 dtype.
     *
     * @param n number of elements (non-negative)
     * @return a lazy range tensor
     */
    static Tensor iota(long n) {
        return TensorFactory.iota(n);
    }

    /**
     * Creates a 1D range tensor [0, 1, 2, ..., n-1] and casts to the target type.
     *
     * @param n number of elements (non-negative)
     * @param dataType target data type (integral or floating-point)
     * @return a lazy range tensor cast to the target type
     */
    static Tensor iota(long n, DataType dataType) {
        return TensorFactory.iota(n, dataType);
    }

    // endregion Constant / Fill Creation
    // region Random Creation

    /**
     * Sets the thread-local RNG seed used by random methods that do not take an explicit {@link
     * RandomKey}.
     */
    static void manualSeed(long seed) {
        TensorFactory.manualSeed(seed);
    }

    /** Alias of {@link #uniform(long, double, double, DataType)} with [0, 1) bounds. */
    static Tensor rand(long size, DataType dataType) {
        return TensorFactory.rand(size, dataType);
    }

    /** Alias of {@link #uniform(long, double, double, DataType, RandomKey)} with [0, 1) bounds. */
    static Tensor rand(long size, DataType dataType, RandomKey randomKey) {
        return TensorFactory.rand(size, dataType, randomKey);
    }

    /** Alias of {@link #uniform(Shape, double, double, DataType)} with [0, 1) bounds. */
    static Tensor rand(Shape shape, DataType dataType) {
        return TensorFactory.rand(shape, dataType);
    }

    /** Alias of {@link #uniform(Shape, double, double, DataType, RandomKey)} with [0, 1) bounds. */
    static Tensor rand(Shape shape, DataType dataType, RandomKey randomKey) {
        return TensorFactory.rand(shape, dataType, randomKey);
    }

    /** Alias of {@link #normal(long, double, double, DataType)} with mean=0 and std=1. */
    static Tensor randn(long size, DataType dataType) {
        return TensorFactory.randn(size, dataType);
    }

    /**
     * Alias of {@link #normal(long, double, double, DataType, RandomKey)} with mean=0 and std=1.
     */
    static Tensor randn(long size, DataType dataType, RandomKey randomKey) {
        return TensorFactory.randn(size, dataType, randomKey);
    }

    /** Alias of {@link #normal(Shape, double, double, DataType)} with mean=0 and std=1. */
    static Tensor randn(Shape shape, DataType dataType) {
        return TensorFactory.randn(shape, dataType);
    }

    /**
     * Alias of {@link #normal(Shape, double, double, DataType, RandomKey)} with mean=0 and std=1.
     */
    static Tensor randn(Shape shape, DataType dataType, RandomKey randomKey) {
        return TensorFactory.randn(shape, dataType, randomKey);
    }

    /** Alias of {@link #uniformInt(long, long, long, DataType)}. */
    static Tensor randInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return TensorFactory.randInt(startInclusive, endExclusive, size, dataType);
    }

    /** Alias of {@link #uniformInt(long, long, long, DataType, RandomKey)}. */
    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.randInt(startInclusive, endExclusive, size, dataType, randomKey);
    }

    /** Alias of {@link #uniformInt(long, long, Shape, DataType)}. */
    static Tensor randInt(long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        return TensorFactory.randInt(startInclusive, endExclusive, shape, dataType);
    }

    /** Alias of {@link #uniformInt(long, long, Shape, DataType, RandomKey)}. */
    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.randInt(startInclusive, endExclusive, shape, dataType, randomKey);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            long size, double startInclusive, double endExclusive, DataType dataType) {
        return TensorFactory.uniform(size, startInclusive, endExclusive, dataType);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            long size,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.uniform(size, startInclusive, endExclusive, dataType, randomKey);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            Shape shape, double startInclusive, double endExclusive, DataType dataType) {
        return TensorFactory.uniform(shape, startInclusive, endExclusive, dataType);
    }

    /** Creates lazy floating-point uniform values in [startInclusive, endExclusive). */
    static Tensor uniform(
            Shape shape,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.uniform(shape, startInclusive, endExclusive, dataType, randomKey);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(long size, double mean, double std, DataType dataType) {
        return TensorFactory.normal(size, mean, std, dataType);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        return TensorFactory.normal(size, mean, std, dataType, randomKey);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(Shape shape, double mean, double std, DataType dataType) {
        return TensorFactory.normal(shape, mean, std, dataType);
    }

    /** Creates lazy standard normal values and applies mean/std affine transform. */
    static Tensor normal(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        return TensorFactory.normal(shape, mean, std, dataType, randomKey);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return TensorFactory.uniformInt(startInclusive, endExclusive, size, dataType);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.uniformInt(startInclusive, endExclusive, size, dataType, randomKey);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        return TensorFactory.uniformInt(startInclusive, endExclusive, shape, dataType);
    }

    /** Creates lazy integral uniform values in [startInclusive, endExclusive). */
    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        return TensorFactory.uniformInt(startInclusive, endExclusive, shape, dataType, randomKey);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(long size, double mean, double std, DataType dataType) {
        return TensorFactory.normalInt(size, mean, std, dataType);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        return TensorFactory.normalInt(size, mean, std, dataType, randomKey);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(Shape shape, double mean, double std, DataType dataType) {
        return TensorFactory.normalInt(shape, mean, std, dataType);
    }

    /** Creates integer-valued normal samples by rounding and clamping to target integer dtype. */
    static Tensor normalInt(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        return TensorFactory.normalInt(shape, mean, std, dataType, randomKey);
    }

    // endregion Random Creation
    // region Fill / Scalar Creation

    /**
     * Creates a tensor filled with the specified float value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (FP32)
     */
    static Tensor full(float value, Shape shape) {
        return TensorFactory.full(value, shape);
    }

    /**
     * Creates a tensor filled with the specified double value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (FP64)
     */
    static Tensor full(double value, Shape shape) {
        return TensorFactory.full(value, shape);
    }

    /**
     * Creates a tensor filled with the specified long value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (I64)
     */
    static Tensor full(long value, Shape shape) {
        return TensorFactory.full(value, shape);
    }

    /**
     * Creates a tensor filled with the specified int value.
     *
     * @param value the fill value
     * @param shape the shape of the tensor
     * @return a lazy tensor filled with the value (I32)
     */
    static Tensor full(int value, Shape shape) {
        return TensorFactory.full(value, shape);
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
        return TensorFactory.full(value, dtype, shape);
    }

    static Tensor broadcasted(int value, Shape shape) {
        return TensorFactory.broadcasted(value, shape);
    }

    static Tensor broadcasted(long value, Shape shape) {
        return TensorFactory.broadcasted(value, shape);
    }

    static Tensor broadcasted(float value, Shape shape) {
        return TensorFactory.broadcasted(value, shape);
    }

    static Tensor broadcasted(double value, Shape shape) {
        return TensorFactory.broadcasted(value, shape);
    }

    static Tensor scalar(int value) {
        return TensorFactory.scalar(value);
    }

    static Tensor scalar(float value) {
        return TensorFactory.scalar(value);
    }

    static Tensor scalar(double value) {
        return TensorFactory.scalar(value);
    }

    static Tensor scalar(long value) {
        return TensorFactory.scalar(value);
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
        return TensorFactory.scalar(value, dtype);
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
        return TensorFactory.scalar(value, dtype);
    }

    // endregion Fill / Scalar Creation
    // region Lazy / Array-backed Creation

    /**
     * Creates a tensor from a float array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and FP32 dtype
     */
    static Tensor of(float[] data) {
        return TensorFactory.of(data);
    }

    /**
     * Creates a tensor from a float array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with FP32 dtype
     */
    static Tensor of(float[] data, Shape shape) {
        return TensorFactory.of(data, shape);
    }

    /**
     * Creates a tensor from a double array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and FP64 dtype
     */
    static Tensor of(double[] data) {
        return TensorFactory.of(data);
    }

    /**
     * Creates a tensor from a double array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with FP64 dtype
     */
    static Tensor of(double[] data, Shape shape) {
        return TensorFactory.of(data, shape);
    }

    /**
     * Creates a tensor from an int array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and I32 dtype
     */
    static Tensor of(int[] data) {
        return TensorFactory.of(data);
    }

    /**
     * Creates a tensor from an int array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with I32 dtype
     */
    static Tensor of(int[] data, Shape shape) {
        return TensorFactory.of(data, shape);
    }

    /**
     * Creates a tensor from a long array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and I64 dtype
     */
    static Tensor of(long[] data) {
        return TensorFactory.of(data);
    }

    /**
     * Creates a tensor from a long array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with I64 dtype
     */
    static Tensor of(long[] data, Shape shape) {
        return TensorFactory.of(data, shape);
    }

    /**
     * Creates a tensor from a boolean array.
     *
     * @param data the source array (copied, not referenced)
     * @return a materialized tensor with shape [data.length] and BOOL dtype
     */
    static Tensor of(boolean[] data) {
        return TensorFactory.of(data);
    }

    /**
     * Creates a tensor from a boolean array with the specified shape.
     *
     * @param data the source array (copied, not referenced)
     * @param shape the shape (must match array length)
     * @return a materialized tensor with BOOL dtype
     */
    static Tensor of(boolean[] data, Shape shape) {
        return TensorFactory.of(data, shape);
    }

    // endregion Lazy / Array-backed Creation
    // endregion Static Creation Methods

}

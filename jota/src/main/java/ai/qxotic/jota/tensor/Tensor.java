package ai.qxotic.jota.tensor;

import ai.qxotic.jota.*;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

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
        return size() == 1;
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

    boolean isMaterialized();

    boolean isLazy();

    MemoryView<?> materialize();

    Optional<MemoryView<?>> tryGetMaterialized();

    Optional<LazyComputation> computation();

    default Tensor add(Tensor other) {
        return TensorOpsContext.require().add(this, other);
    }

    default Tensor add(Number scalar) {
        return TensorOpsContext.require().add(this, scalar);
    }

    default Tensor subtract(Tensor other) {
        return TensorOpsContext.require().subtract(this, other);
    }

    default Tensor subtract(Number scalar) {
        return TensorOpsContext.require().subtract(this, scalar);
    }

    default Tensor multiply(Tensor other) {
        return TensorOpsContext.require().multiply(this, other);
    }

    default Tensor multiply(Number scalar) {
        return TensorOpsContext.require().multiply(this, scalar);
    }

    default Tensor divide(Tensor other) {
        return TensorOpsContext.require().divide(this, other);
    }

    default Tensor divide(Number scalar) {
        return TensorOpsContext.require().divide(this, scalar);
    }

    default Tensor min(Tensor other) {
        return TensorOpsContext.require().min(this, other);
    }

    default Tensor min() {
        return TensorOpsContext.require().min(this);
    }

    default Tensor min(int _axis, int... _axes) {
        return TensorOpsContext.require().min(this, _axis, _axes);
    }

    default Tensor min(boolean keepDims, int _axis, int... _axes) {
        return TensorOpsContext.require().min(this, keepDims, _axis, _axes);
    }

    default Tensor max(Tensor other) {
        return TensorOpsContext.require().max(this, other);
    }

    default Tensor max() {
        return TensorOpsContext.require().max(this);
    }

    default Tensor max(int _axis, int... _axes) {
        return TensorOpsContext.require().max(this, _axis, _axes);
    }

    default Tensor max(boolean keepDims, int _axis, int... _axes) {
        return TensorOpsContext.require().max(this, keepDims, _axis, _axes);
    }

    default Tensor any() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        return TensorOpsContext.require().max(this);
    }

    default Tensor all() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        return TensorOpsContext.require().min(this);
    }

    default Tensor to(Device device) {
        return TensorOpsContext.require().to(this, device);
    }

    default Tensor contiguous() {
        return TensorOpsContext.require().contiguous(this);
    }

    default Tensor bitwiseNot() {
        return TensorOpsContext.require().bitwiseNot(this);
    }

    default Tensor bitwiseAnd(Tensor other) {
        return TensorOpsContext.require().bitwiseAnd(this, other);
    }

    default Tensor bitwiseOr(Tensor other) {
        return TensorOpsContext.require().bitwiseOr(this, other);
    }

    default Tensor bitwiseXor(Tensor other) {
        return TensorOpsContext.require().bitwiseXor(this, other);
    }

    default Tensor logicalNot() {
        return TensorOpsContext.require().logicalNot(this);
    }

    default Tensor logicalAnd(Tensor other) {
        return TensorOpsContext.require().logicalAnd(this, other);
    }

    default Tensor logicalOr(Tensor other) {
        return TensorOpsContext.require().logicalOr(this, other);
    }

    default Tensor logicalXor(Tensor other) {
        return TensorOpsContext.require().logicalXor(this, other);
    }

    default Tensor equal(Tensor other) {
        return TensorOpsContext.require().equal(this, other);
    }

    default Tensor lessThan(Tensor other) {
        return TensorOpsContext.require().lessThan(this, other);
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

    static Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        return TensorOpsContext.require().where(condition, trueValue, falseValue);
    }

    default Tensor sum(DataType accumulatorType) {
        return TensorOpsContext.require().sum(this, accumulatorType);
    }

    default Tensor sum(DataType accumulatorType, int _axis, int... _axes) {
        return TensorOpsContext.require().sum(this, accumulatorType, _axis, _axes);
    }

    default Tensor sum(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return TensorOpsContext.require().sum(this, accumulatorType, keepDims, _axis, _axes);
    }

    default Tensor product(DataType accumulatorType) {
        return TensorOpsContext.require().product(this, accumulatorType);
    }

    default Tensor product(DataType accumulatorType, int _axis, int... _axes) {
        return TensorOpsContext.require().product(this, accumulatorType, _axis, _axes);
    }

    default Tensor product(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return TensorOpsContext.require().product(this, accumulatorType, keepDims, _axis, _axes);
    }

    default Tensor cast(DataType targetType) {
        return TensorOpsContext.require().cast(this, targetType);
    }

    default Tensor negate() {
        return TensorOpsContext.require().negate(this);
    }

    default Tensor abs() {
        return TensorOpsContext.require().abs(this);
    }

    default Tensor exp() {
        return TensorOpsContext.require().exp(this);
    }

    default Tensor log() {
        return TensorOpsContext.require().log(this);
    }

    default Tensor sqrt() {
        return TensorOpsContext.require().sqrt(this);
    }

    default Tensor square() {
        return TensorOpsContext.require().square(this);
    }

    default Tensor sin() {
        return TensorOpsContext.require().sin(this);
    }

    default Tensor cos() {
        return TensorOpsContext.require().cos(this);
    }

    default Tensor tanh() {
        return TensorOpsContext.require().tanh(this);
    }

    default Tensor sigmoid() {
        return TensorOpsContext.require().sigmoid(this);
    }

    default Tensor relu() {
        return TensorOpsContext.require().relu(this);
    }

    default Tensor gelu() {
        return TensorOpsContext.require().gelu(this);
    }

    default Tensor silu() {
        return TensorOpsContext.require().silu(this);
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

    static Tensor broadcasted(float value, Shape shape) {
        return broadcasted(Float.valueOf(value), DataType.FP32, shape, Device.defaultDevice());
    }

    static Tensor broadcasted(long value, Shape shape) {
        return broadcasted(Long.valueOf(value), DataType.I64, shape, Device.defaultDevice());
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

    static Tensor lazy(LazyComputation computation, DataType dtype, Layout layout, Device device) {
        return new LazyTensor(computation, dtype, layout, device);
    }

    private static Tensor broadcasted(Number value, DataType dataType, Shape shape, Device device) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(device, "device");
        Layout layout = Layout.of(shape, Stride.zeros(shape));
        ConstantComputation computation = ConstantComputation.of(value, dataType, shape, device);
        return lazy(computation, dataType, layout, device);
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
        MemoryContext<?> context = Environment.current().registry().context(Device.defaultDevice());
        MemoryView<?> view = copyFloatArray(context, data, shape);
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
        MemoryContext<?> context = Environment.current().registry().context(Device.defaultDevice());
        MemoryView<?> view = copyDoubleArray(context, data, shape);
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
        MemoryContext<?> context = Environment.current().registry().context(Device.defaultDevice());
        MemoryView<?> view = copyIntArray(context, data, shape);
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
        MemoryContext<?> context = Environment.current().registry().context(Device.defaultDevice());
        MemoryView<?> view = copyLongArray(context, data, shape);
        return of(view);
    }

    private static <B> MemoryView<B> copyFloatArray(
            MemoryContext<B> context, float[] data, Shape shape) {
        Memory<B> dst = context.memoryAllocator().allocateMemory(DataType.FP32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps =
                ContextFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                context.memoryOperations(),
                dst,
                0,
                (long) data.length * Float.BYTES);
        return MemoryView.of(dst, DataType.FP32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyDoubleArray(
            MemoryContext<B> context, double[] data, Shape shape) {
        Memory<B> dst = context.memoryAllocator().allocateMemory(DataType.FP64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps =
                ContextFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                context.memoryOperations(),
                dst,
                0,
                (long) data.length * Double.BYTES);
        return MemoryView.of(dst, DataType.FP64, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyIntArray(
            MemoryContext<B> context, int[] data, Shape shape) {
        Memory<B> dst = context.memoryAllocator().allocateMemory(DataType.I32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps =
                ContextFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                context.memoryOperations(),
                dst,
                0,
                (long) data.length * Integer.BYTES);
        return MemoryView.of(dst, DataType.I32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyLongArray(
            MemoryContext<B> context, long[] data, Shape shape) {
        Memory<B> dst = context.memoryAllocator().allocateMemory(DataType.I64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps =
                ContextFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                context.memoryOperations(),
                dst,
                0,
                (long) data.length * Long.BYTES);
        return MemoryView.of(dst, DataType.I64, Layout.rowMajor(shape));
    }

    @Deprecated
    default Tensor realize() {
        materialize();
        return this;
    }
}

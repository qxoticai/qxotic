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
        return tryFoldBinaryOp(other, BinaryOp.ADD)
                .orElseGet(() -> TensorOpsContext.require().add(this, broadcastIfScalar(other)));
    }

    default Tensor add(int scalar) {
        return add(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor add(long scalar) {
        return add(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor add(float scalar) {
        return add(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor add(double scalar) {
        return add(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor subtract(Tensor other) {
        return tryFoldBinaryOp(other, BinaryOp.SUBTRACT)
                .orElseGet(
                        () -> TensorOpsContext.require().subtract(this, broadcastIfScalar(other)));
    }

    default Tensor subtract(int scalar) {
        return subtract(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor subtract(long scalar) {
        return subtract(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor subtract(float scalar) {
        return subtract(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor subtract(double scalar) {
        return subtract(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor multiply(Tensor other) {
        return tryFoldBinaryOp(other, BinaryOp.MULTIPLY)
                .orElseGet(
                        () -> TensorOpsContext.require().multiply(this, broadcastIfScalar(other)));
    }

    default Tensor multiply(int scalar) {
        return multiply(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor multiply(long scalar) {
        return multiply(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor multiply(float scalar) {
        return multiply(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor multiply(double scalar) {
        return multiply(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor divide(Tensor other) {
        return tryFoldBinaryOp(other, BinaryOp.DIVIDE)
                .orElseGet(() -> TensorOpsContext.require().divide(this, broadcastIfScalar(other)));
    }

    default Tensor divide(int scalar) {
        return divide(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor divide(long scalar) {
        return divide(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor divide(float scalar) {
        return divide(Tensor.broadcasted(scalar, shape()));
    }

    default Tensor divide(double scalar) {
        return divide(Tensor.broadcasted(scalar, shape()));
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
        return TensorOpsContext.require().view(this, newShape);
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
        return TensorOpsContext.require().reshape(this, newShape);
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
        if (this.dataType() == targetType) {
            return this;
        }
        return TensorOpsContext.require().cast(this, targetType);
    }

    default Tensor negate() {
        return tryFoldUnaryOp(UnaryOp.NEGATE)
                .orElseGet(() -> TensorOpsContext.require().negate(this));
    }

    default Tensor abs() {
        return tryFoldUnaryOp(UnaryOp.ABS).orElseGet(() -> TensorOpsContext.require().abs(this));
    }

    default Tensor exp() {
        return tryFoldUnaryOp(UnaryOp.EXP).orElseGet(() -> TensorOpsContext.require().exp(this));
    }

    default Tensor log() {
        return tryFoldUnaryOp(UnaryOp.LOG).orElseGet(() -> TensorOpsContext.require().log(this));
    }

    default Tensor sqrt() {
        return tryFoldUnaryOp(UnaryOp.SQRT).orElseGet(() -> TensorOpsContext.require().sqrt(this));
    }

    default Tensor square() {
        return tryFoldUnaryOp(UnaryOp.SQUARE)
                .orElseGet(() -> TensorOpsContext.require().square(this));
    }

    default Tensor sin() {
        return tryFoldUnaryOp(UnaryOp.SIN).orElseGet(() -> TensorOpsContext.require().sin(this));
    }

    default Tensor cos() {
        return tryFoldUnaryOp(UnaryOp.COS).orElseGet(() -> TensorOpsContext.require().cos(this));
    }

    default Tensor tanh() {
        return tryFoldUnaryOp(UnaryOp.TANH).orElseGet(() -> TensorOpsContext.require().tanh(this));
    }

    default Tensor relu() {
        requireFloatingPoint("relu");
        return max(Tensor.full(0f, dataType(), shape()));
    }

    default Tensor sigmoid() {
        requireFloatingPoint("sigmoid");
        return negate().exp().add(Tensor.scalar(1, dataType())).reciprocal();
    }

    default Tensor silu() {
        requireFloatingPoint("silu");
        return multiply(sigmoid()); // x * sigmoid(x)
    }

    default Tensor gelu() {
        requireFloatingPoint("gelu");
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

    private void requireFloatingPoint(String op) {
        if (!dataType().isFloatingPoint()) {
            throw new IllegalArgumentException(
                    op + " requires floating-point tensor, got " + dataType());
        }
    }

    private Tensor broadcastIfScalar(Tensor other) {
        if (!other.isScalar() || this.isScalar()) {
            return other;
        }
        // Extract value from scalar and broadcast to this shape
        return other.computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .map(c -> Tensor.full(c.value(), c.dataType(), shape()))
                .orElse(other);
    }

    private Optional<ConstantComputation> asConstant() {
        return computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast);
    }

    private Optional<Tensor> tryFoldBinaryOp(Tensor other, BinaryOp op) {
        Optional<ConstantComputation> leftConst = this.asConstant();
        Optional<ConstantComputation> rightConst = other.asConstant();
        if (leftConst.isEmpty() || rightConst.isEmpty()) {
            return Optional.empty();
        }
        ConstantComputation left = leftConst.get();
        ConstantComputation right = rightConst.get();
        DataType resultType = TypeRules.promote(left.dataType(), right.dataType());
        double a = left.value().doubleValue();
        double b = right.value().doubleValue();
        double result;
        if (op == BinaryOp.ADD) {
            result = a + b;
        } else if (op == BinaryOp.SUBTRACT) {
            result = a - b;
        } else if (op == BinaryOp.MULTIPLY) {
            result = a * b;
        } else if (op == BinaryOp.DIVIDE) {
            result = a / b;
        } else if (op == BinaryOp.MIN) {
            result = Math.min(a, b);
        } else if (op == BinaryOp.MAX) {
            result = Math.max(a, b);
        } else {
            throw new UnsupportedOperationException("Cannot fold: " + op);
        }
        return Optional.of(Tensor.scalar(result, resultType));
    }

    private Optional<Tensor> tryFoldUnaryOp(UnaryOp op) {
        Optional<ConstantComputation> constant = this.asConstant();
        if (constant.isEmpty()) {
            return Optional.empty();
        }
        ConstantComputation c = constant.get();
        double a = c.value().doubleValue();
        double result;
        if (op == UnaryOp.NEGATE) {
            result = -a;
        } else if (op == UnaryOp.ABS) {
            result = Math.abs(a);
        } else if (op == UnaryOp.EXP) {
            result = Math.exp(a);
        } else if (op == UnaryOp.LOG) {
            result = Math.log(a);
        } else if (op == UnaryOp.SQRT) {
            result = Math.sqrt(a);
        } else if (op == UnaryOp.SQUARE) {
            result = a * a;
        } else if (op == UnaryOp.SIN) {
            result = Math.sin(a);
        } else if (op == UnaryOp.COS) {
            result = Math.cos(a);
        } else if (op == UnaryOp.TANH) {
            result = Math.tanh(a);
        } else if (op == UnaryOp.RECIPROCAL) {
            result = 1.0 / a;
        } else {
            return Optional.empty();
        }
        return Optional.of(Tensor.scalar(result, c.dataType()));
    }

    default Tensor reciprocal() {
        if (!dataType().isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "reciprocal requires floating-point tensor, got " + dataType());
        }
        return tryFoldUnaryOp(UnaryOp.RECIPROCAL)
                .orElseGet(() -> TensorOpsContext.require().reciprocal(this));
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

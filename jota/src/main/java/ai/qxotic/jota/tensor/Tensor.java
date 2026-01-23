package ai.qxotic.jota.tensor;

import ai.qxotic.jota.*;
import ai.qxotic.jota.memory.MemoryView;
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

    @Deprecated
    default Tensor realize() {
        materialize();
        return this;
    }
}

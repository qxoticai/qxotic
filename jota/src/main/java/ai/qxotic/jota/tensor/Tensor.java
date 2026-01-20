package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
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

    default Tensor copyTo(Device targetDevice) {
        MemoryView<?> srcView = materialize();
        MemoryContext<?> srcContext = DeviceRegistry.context(device());
        MemoryContext<?> dstContext = DeviceRegistry.context(targetDevice);
        @SuppressWarnings("unchecked")
        MemoryView<Object> dstView =
                (MemoryView<Object>)
                        MemoryView.of(
                                dstContext.memoryAllocator().allocateMemory(dataType(), layout().shape()),
                                dataType(),
                                layout());
        @SuppressWarnings("unchecked")
        MemoryContext<Object> castSrcContext = (MemoryContext<Object>) srcContext;
        @SuppressWarnings("unchecked")
        MemoryContext<Object> castDstContext = (MemoryContext<Object>) dstContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> castSrcView = (MemoryView<Object>) srcView;
        MemoryContext.copy(castSrcContext, castSrcView, castDstContext, dstView);
        return Tensor.of(dstView);
    }


    default boolean isScalar() {
        return size() == 1;
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

    default Tensor sum(DataType accumulatorType) {
        return TensorOpsContext.require().sum(this, accumulatorType);
    }

    default Tensor sum(DataType accumulatorType, int _axis, int... _axes) {
        return TensorOpsContext.require().sum(this, accumulatorType, _axis, _axes);
    }

    default Tensor sum(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return TensorOpsContext.require().sum(this, accumulatorType, keepDims, _axis, _axes);
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

    static Tensor lazy(LazyComputation computation, DataType dtype, Layout layout, Device device) {
        return new LazyTensor(computation, dtype, layout, device);
    }

    @Deprecated
    default Tensor realize() {
        materialize();
        return this;
    }
}

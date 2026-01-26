package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import java.util.function.BiFunction;
import java.util.function.Function;

final class LazyTensorOps implements TensorOps {

    @Override
    public MemoryContext<?> context() {
        throw new UnsupportedOperationException("Lazy ops do not expose a memory context");
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().add(t0, t1));
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().subtract(t0, t1));
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().multiply(t0, t1));
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().divide(t0, t1));
    }

    @Override
    public Tensor min(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().min(t0, t1));
    }

    @Override
    public Tensor max(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().max(t0, t1));
    }

    @Override
    public Tensor negate(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().negate(t));
    }

    @Override
    public Tensor abs(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().abs(t));
    }

    @Override
    public Tensor exp(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().exp(t));
    }

    @Override
    public Tensor log(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().log(t));
    }

    @Override
    public Tensor sqrt(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().sqrt(t));
    }

    @Override
    public Tensor sin(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().sin(t));
    }

    @Override
    public Tensor cos(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().cos(t));
    }

    @Override
    public Tensor tanh(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().tanh(t));
    }

    @Override
    public Tensor reciprocal(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().reciprocal(t));
    }

    @Override
    public Tensor to(Tensor x, Device device) {
        return traceUnary(x, t -> TensorOpsContext.require().to(t, device));
    }

    @Override
    public Tensor contiguous(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().contiguous(t));
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().bitwiseNot(t));
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().bitwiseAnd(t0, t1));
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().bitwiseOr(t0, t1));
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().bitwiseXor(t0, t1));
    }

    @Override
    public Tensor logicalNot(Tensor x) {
        return traceUnary(x, t -> TensorOpsContext.require().logicalNot(t));
    }

    @Override
    public Tensor logicalAnd(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().logicalAnd(t0, t1));
    }

    @Override
    public Tensor logicalOr(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().logicalOr(t0, t1));
    }

    @Override
    public Tensor logicalXor(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().logicalXor(t0, t1));
    }

    @Override
    public Tensor equal(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().equal(t0, t1));
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().lessThan(t0, t1));
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        return traceTernary(
                condition,
                trueValue,
                falseValue,
                (c, t, f) -> TensorOpsContext.require().where(c, t, f));
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return traceUnary(
                x, t -> TensorOpsContext.require().sum(t, accumulatorType, keepDims, _axis, _axes));
    }

    @Override
    public Tensor product(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        return traceUnary(
                x,
                t ->
                        TensorOpsContext.require()
                                .product(t, accumulatorType, keepDims, _axis, _axes));
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        return traceUnary(x, t -> TensorOpsContext.require().mean(t, axis, keepDims));
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return traceUnary(x, t -> TensorOpsContext.require().max(t));
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        return traceUnary(x, t -> TensorOpsContext.require().min(t));
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().matmul(t0, t1));
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        return traceBinary(a, b, (t0, t1) -> TensorOpsContext.require().batchedMatmul(t0, t1));
    }

    @Override
    public Tensor viewTransform(Tensor x, Layout layout, long byteOffsetDelta, String hint) {
        return traceUnary(
                x, t -> TensorOpsContext.require().viewTransform(t, layout, byteOffsetDelta, hint));
    }

    @Override
    public Tensor reshape(Tensor x, Shape newShape) {
        return traceUnary(x, t -> TensorOpsContext.require().reshape(t, newShape));
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        return traceUnary(x, t -> TensorOpsContext.require().cast(t, targetType));
    }

    private Tensor traceUnary(Tensor x, Function<Tensor, Tensor> fn) {
        return Tracer.trace(x, fn);
    }

    private Tensor traceBinary(Tensor a, Tensor b, BiFunction<Tensor, Tensor, Tensor> fn) {
        return Tracer.trace(a, b, fn);
    }

    private Tensor traceTernary(
            Tensor a, Tensor b, Tensor c, TriFunction<Tensor, Tensor, Tensor, Tensor> fn) {
        return Tracer.trace(a, b, c, fn);
    }
}

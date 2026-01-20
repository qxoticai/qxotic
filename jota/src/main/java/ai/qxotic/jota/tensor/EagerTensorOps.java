package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Objects;

public class EagerTensorOps implements TensorOps {

    private final MemoryContext<?> context;

    public EagerTensorOps(MemoryContext<?> context) {
        this.context = Objects.requireNonNull(context);
    }

    @Override
    public MemoryContext<?> context() {
        return context;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.ADD);
    }

    @Override
    public Tensor subtract(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.SUBTRACT);
    }

    @Override
    public Tensor multiply(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MULTIPLY);
    }

    @Override
    public Tensor divide(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.DIVIDE);
    }

    @Override
    public Tensor min(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MIN);
    }

    @Override
    public Tensor max(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.MAX);
    }

    @Override
    public Tensor add(Tensor a, Number scalar) {
        return scalarOp(a, scalar, BinaryOp.ADD);
    }

    @Override
    public Tensor subtract(Tensor a, Number scalar) {
        return scalarOp(a, scalar, BinaryOp.SUBTRACT);
    }

    @Override
    public Tensor multiply(Tensor a, Number scalar) {
        return scalarOp(a, scalar, BinaryOp.MULTIPLY);
    }

    @Override
    public Tensor divide(Tensor a, Number scalar) {
        return scalarOp(a, scalar, BinaryOp.DIVIDE);
    }

    @Override
    public Tensor negate(Tensor x) {
        return unaryOp(x, UnaryOp.NEGATE);
    }

    @Override
    public Tensor abs(Tensor x) {
        return unaryOp(x, UnaryOp.ABS);
    }

    @Override
    public Tensor exp(Tensor x) {
        return unaryOp(x, UnaryOp.EXP);
    }

    @Override
    public Tensor log(Tensor x) {
        return unaryOp(x, UnaryOp.LOG);
    }

    @Override
    public Tensor sqrt(Tensor x) {
        return unaryOp(x, UnaryOp.SQRT);
    }

    @Override
    public Tensor square(Tensor x) {
        return unaryOp(x, UnaryOp.SQUARE);
    }

    @Override
    public Tensor sin(Tensor x) {
        return unaryOp(x, UnaryOp.SIN);
    }

    @Override
    public Tensor cos(Tensor x) {
        return unaryOp(x, UnaryOp.COS);
    }

    @Override
    public Tensor tanh(Tensor x) {
        return unaryOp(x, UnaryOp.TANH);
    }

    @Override
    public Tensor sigmoid(Tensor x) {
        return unaryOp(x, UnaryOp.SIGMOID);
    }

    @Override
    public Tensor relu(Tensor x) {
        return unaryOp(x, UnaryOp.RELU);
    }

    @Override
    public Tensor gelu(Tensor x) {
        return unaryOp(x, UnaryOp.GELU);
    }

    @Override
    public Tensor silu(Tensor x) {
        return unaryOp(x, UnaryOp.SILU);
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor mean(Tensor x, int axis, boolean keepDims) {
        return reductionOp(x, ReductionOp.MEAN, axis, keepDims);
    }

    @Override
    public Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor meanAll(Tensor x) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor maxAll(Tensor x) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor minAll(Tensor x) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor transpose(Tensor x, int axis0, int axis1) {
        MemoryView<?> view = x.materialize();
        MemoryView<?> transposed = view.transpose(axis0, axis1);
        return Tensor.of(transposed);
    }

    @Override
    public Tensor reshape(Tensor x, Shape newShape) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor view(Tensor x, Shape newShape) {
        MemoryView<?> memView = x.materialize();
        MemoryView<?> reshaped = memView.view(newShape);
        return Tensor.of(reshaped);
    }

    @Override
    public Tensor broadcast(Tensor x, Shape targetShape) {
        MemoryView<?> view = x.materialize();
        MemoryView<?> broadcasted = view.broadcast(targetShape);
        return Tensor.of(broadcasted);
    }

    @Override
    public Tensor expand(Tensor x, Shape targetShape) {
        MemoryView<?> view = x.materialize();
        MemoryView<?> expanded = view.expand(targetShape);
        return Tensor.of(expanded);
    }

    @Override
    public Tensor slice(Tensor x, int axis, long start, long end) {
        MemoryView<?> view = x.materialize();
        MemoryView<?> sliced = view.slice(axis, start, end);
        return Tensor.of(sliced);
    }

    @Override
    public Tensor softmax(Tensor x, int axis) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor layerNorm(Tensor x, Tensor weight, Tensor bias, float eps) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        return Tracer.trace(x, input -> input.cast(targetType));
    }

    @Override
    public Tensor quantize(Tensor x, DataType quantType) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor dequantize(Tensor x, DataType targetType) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor copy(Tensor x) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor clone(Tensor x) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void copyInto(Tensor src, Tensor dst) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    private MemoryView<?> allocate(DataType dtype, Shape shape) {
        Layout layout = Layout.rowMajor(shape);
        return MemoryView.of(
                context.memoryAllocator().allocateMemory(dtype.byteSizeFor(shape)), dtype, layout);
    }

    private Tensor unaryOp(Tensor x, UnaryOp op) {
        throw new UnsupportedOperationException("Generic unary op dispatch not yet implemented");
    }

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOp op) {
        throw new UnsupportedOperationException("Generic binary op dispatch not yet implemented");
    }

    private Tensor scalarOp(Tensor a, Number scalar, BinaryOp op) {
        throw new UnsupportedOperationException("Generic scalar op dispatch not yet implemented");
    }

    private Tensor reductionOp(Tensor x, ReductionOp op, int axis, boolean keepDims) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }
}

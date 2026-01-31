package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.impl.ViewTransforms;
import ai.qxotic.jota.ir.tir.ViewKind;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
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
    public Tensor reciprocal(Tensor x) {
        return unaryOp(x, UnaryOp.RECIPROCAL);
    }

    @Override
    public Tensor to(Tensor x, Device device) {
        Objects.requireNonNull(device, "device");
        if (x.device().equals(device)) {
            return x;
        }
        MemoryView<?> sourceView = x.materialize();
        MemoryContext<?> sourceContext = Environment.current().backend(x.device()).memoryContext();
        MemoryContext<?> targetContext = Environment.current().backend(device).memoryContext();
        if (!targetContext.supportsDataType(sourceView.dataType())) {
            throw new IllegalArgumentException(
                    "Target context does not support data type: " + sourceView.dataType());
        }
        if (!sourceContext.device().equals(targetContext.device()) && !sourceView.isContiguous()) {
            throw new IllegalArgumentException(
                    "Target backend cannot preserve layout; call x.contiguous().to(device)");
        }
        OutputBufferSpec outputSpec = computeOutputSpec(sourceView.layout(), sourceView.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        targetContext.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        sourceView.dataType(),
                        sourceView.layout());
        copyBetweenContexts(sourceContext, sourceView, targetContext, targetView);
        return Tensor.of(targetView);
    }

    @Override
    public Tensor contiguous(Tensor x) {
        MemoryView<?> sourceView = x.materialize();
        if (sourceView.isContiguous()) {
            return x;
        }
        boolean primitive =
                sourceView.dataType() == DataType.BOOL
                        || sourceView.dataType() == DataType.I8
                        || sourceView.dataType() == DataType.I16
                        || sourceView.dataType() == DataType.I32
                        || sourceView.dataType() == DataType.I64
                        || sourceView.dataType() == DataType.FP16
                        || sourceView.dataType() == DataType.BF16
                        || sourceView.dataType() == DataType.FP32
                        || sourceView.dataType() == DataType.FP64;
        if (!primitive) {
            throw new IllegalArgumentException(
                    "contiguous requires primitive data type, got " + sourceView.dataType());
        }
        Layout layout = Layout.rowMajor(sourceView.layout().shape());
        OutputBufferSpec outputSpec = computeOutputSpec(layout, sourceView.dataType());
        MemoryView<?> targetView =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(outputSpec.byteSize),
                        outputSpec.byteOffset,
                        sourceView.dataType(),
                        layout);
        copyBetweenContexts(context, sourceView, context, targetView);
        return Tensor.of(targetView);
    }

    @Override
    public Tensor bitwiseNot(Tensor x) {
        return unaryOp(x, UnaryOp.BITWISE_NOT);
    }

    @Override
    public Tensor bitwiseAnd(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.BITWISE_AND);
    }

    @Override
    public Tensor bitwiseOr(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.BITWISE_OR);
    }

    @Override
    public Tensor bitwiseXor(Tensor a, Tensor b) {
        return binaryOp(a, b, BinaryOp.BITWISE_XOR);
    }

    @Override
    public Tensor logicalNot(Tensor x) {
        return unaryOp(x, UnaryOp.LOGICAL_NOT);
    }

    @Override
    public Tensor logicalAnd(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va && vb);
    }

    @Override
    public Tensor logicalOr(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va || vb);
    }

    @Override
    public Tensor logicalXor(Tensor a, Tensor b) {
        return logicalBinaryOp(a, b, (va, vb) -> va ^ vb);
    }

    @FunctionalInterface
    private interface LogicalBinaryFunc {
        boolean apply(boolean a, boolean b);
    }

    private Tensor logicalBinaryOp(Tensor a, Tensor b, LogicalBinaryFunc func) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        if (viewA.dataType() != DataType.BOOL || viewB.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Logical ops require BOOL tensors");
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(DataType.BOOL, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            boolean va = access.readByte(memA, offA) != 0;
            boolean vb = access.readByte(memB, offB) != 0;
            access.writeByte(memOut, offOut, (byte) (func.apply(va, vb) ? 1 : 0));
        }
        return Tensor.of(output);
    }

    @Override
    public Tensor equal(Tensor a, Tensor b) {
        return compareOp(a, b, BinaryOp.EQUAL);
    }

    @Override
    public Tensor lessThan(Tensor a, Tensor b) {
        return compareOp(a, b, BinaryOp.LESS_THAN);
    }

    private Tensor compareOp(Tensor a, Tensor b, BinaryOp op) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        DataType dtype = viewA.dataType();
        if (dtype != viewB.dataType()) {
            throw new IllegalArgumentException(
                    "Type mismatch: " + dtype + " vs " + viewB.dataType());
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(DataType.BOOL, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            boolean result = applyCompareOp(access, memA, offA, memB, offB, dtype, op);
            access.writeByte(memOut, offOut, (byte) (result ? 1 : 0));
        }
        return Tensor.of(output);
    }

    private boolean applyCompareOp(
            MemoryAccess<Object> access,
            Memory<Object> memA,
            long offA,
            Memory<Object> memB,
            long offB,
            DataType dtype,
            BinaryOp op) {
        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte a = access.readByte(memA, offA);
            byte b = access.readByte(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I16) {
            short a = access.readShort(memA, offA);
            short b = access.readShort(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I32) {
            int a = access.readInt(memA, offA);
            int b = access.readInt(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.I64) {
            long a = access.readLong(memA, offA);
            long b = access.readLong(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.FP32) {
            float a = access.readFloat(memA, offA);
            float b = access.readFloat(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else if (dtype == DataType.FP64) {
            double a = access.readDouble(memA, offA);
            double b = access.readDouble(memB, offB);
            return ConstantFolder.evalCompare(a, b, op);
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        throw new UnsupportedOperationException("Generic where op dispatch not yet implemented");
    }

    @Override
    public Tensor sum(
            Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }

    @Override
    public Tensor product(
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
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Tensor viewTransform(Tensor x, ViewTransforms.ViewTransformSpec spec) {
        // Preserve lazy ConstantComputation for broadcasts/reshapes - don't materialize
        // This is critical for IR tracing to recognize constants
        if (x.computation().orElse(null) instanceof ConstantComputation constComp
                && !x.isMaterialized()) {
            Shape newShape = spec.layout().shape();
            // For constants, we can just create a new lazy constant with the new shape
            // The strides will be all zeros (broadcast semantics)
            return Tensor.lazy(
                    new ConstantComputation(
                            constComp.rawBits(), constComp.dataType(), newShape, constComp.device()),
                    constComp.dataType(),
                    spec.layout(),
                    x.device());
        }

        // Eager path: if lazy indexing is needed, force contiguous copy first
        if (spec.needsLazyIndexing()) {
            Tensor contiguousTensor = contiguous(x);
            // Recompute spec with contiguous layout
            ViewTransforms.ViewTransformSpec newSpec =
                    switch (spec.kind()) {
                        case ViewKind.Reshape reshape ->
                                ViewTransforms.view(contiguousTensor.layout(), reshape.toShape());
                        case ViewKind.Broadcast broadcast ->
                                ViewTransforms.broadcast(
                                        contiguousTensor.layout(), broadcast.toShape());
                        case ViewKind.Expand expand ->
                                ViewTransforms.expand(contiguousTensor.layout(), expand.toShape());
                        case ViewKind.Transpose transpose ->
                                ViewTransforms.permute(
                                        contiguousTensor.layout(), transpose.permutation());
                        case ViewKind.Slice slice ->
                                ViewTransforms.slice(
                                        contiguousTensor.layout(),
                                        x.dataType(),
                                        slice.axis(),
                                        slice.start(),
                                        slice.start() + spec.layout().shape().flatAt(slice.axis()),
                                        slice.step());
                    };
            return viewTransform(contiguousTensor, newSpec);
        }

        MemoryView<?> sourceView = x.materialize();
        MemoryView<?> transformed =
                MemoryView.of(
                        sourceView.memory(),
                        sourceView.byteOffset() + spec.byteOffsetDelta(),
                        sourceView.dataType(),
                        spec.layout());
        return Tensor.of(transformed);
    }

    @Override
    public Tensor reshape(Tensor x, Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(x.layout(), newShape);
        return viewTransform(x, spec);
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        MemoryView<?> view = x.materialize();
        if (view.dataType() == targetType) {
            return x;
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(targetType, view.shape());
        long size = view.shape().size();
        @SuppressWarnings("unchecked")
        Memory<Object> srcMemory = (Memory<Object>) view.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> dstMemory = (Memory<Object>) output.memory();
        DataType srcType = view.dataType();
        for (long i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(view, i);
            long dstOffset = Indexing.linearToOffset(output, i);
            double value = readAsDouble(access, srcMemory, srcOffset, srcType);
            writeFromDouble(access, dstMemory, dstOffset, targetType, value);
        }
        return Tensor.of(output);
    }

    private double readAsDouble(
            MemoryAccess<Object> access, Memory<Object> memory, long offset, DataType type) {
        if (type == DataType.BOOL || type == DataType.I8) {
            return access.readByte(memory, offset);
        } else if (type == DataType.I16 || type == DataType.FP16 || type == DataType.BF16) {
            return access.readShort(memory, offset);
        } else if (type == DataType.I32) {
            return access.readInt(memory, offset);
        } else if (type == DataType.FP32) {
            return access.readFloat(memory, offset);
        } else if (type == DataType.I64) {
            return access.readLong(memory, offset);
        } else if (type == DataType.FP64) {
            return access.readDouble(memory, offset);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + type);
        }
    }

    private void writeFromDouble(
            MemoryAccess<Object> access,
            Memory<Object> memory,
            long offset,
            DataType type,
            double value) {
        if (type == DataType.BOOL) {
            access.writeByte(memory, offset, (byte) (value != 0 ? 1 : 0));
        } else if (type == DataType.I8) {
            access.writeByte(memory, offset, (byte) value);
        } else if (type == DataType.I16) {
            access.writeShort(memory, offset, (short) value);
        } else if (type == DataType.I32) {
            access.writeInt(memory, offset, (int) value);
        } else if (type == DataType.FP32) {
            access.writeFloat(memory, offset, (float) value);
        } else if (type == DataType.I64) {
            access.writeLong(memory, offset, (long) value);
        } else if (type == DataType.FP64) {
            access.writeDouble(memory, offset, value);
        } else if (type == DataType.FP16 || type == DataType.BF16) {
            access.writeShort(memory, offset, (short) Float.floatToFloat16((float) value));
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + type);
        }
    }

    private MemoryView<?> allocate(DataType dtype, Shape shape) {
        Layout layout = Layout.rowMajor(shape);
        return MemoryView.of(
                context.memoryAllocator().allocateMemory(dtype.byteSizeFor(shape)), dtype, layout);
    }

    private Tensor unaryOp(Tensor x, UnaryOp op) {
        MemoryView<?> view = x.materialize();
        DataType dtype = view.dataType();
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, view.shape());
        long size = view.shape().size();
        @SuppressWarnings("unchecked")
        Memory<Object> srcMemory = (Memory<Object>) view.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> dstMemory = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(view, i);
            long dstOffset = Indexing.linearToOffset(output, i);
            applyUnaryOp(access, srcMemory, srcOffset, dstMemory, dstOffset, dtype, op);
        }
        return Tensor.of(output);
    }

    private void applyUnaryOp(
            MemoryAccess<Object> access,
            Memory<Object> srcMem,
            long srcOff,
            Memory<Object> dstMem,
            long dstOff,
            DataType dtype,
            UnaryOp op) {
        if (dtype == DataType.BOOL) {
            byte v = access.readByte(srcMem, srcOff);
            access.writeByte(dstMem, dstOff, ConstantFolder.evalBool(v != 0, op));
        } else if (dtype == DataType.I8) {
            byte v = access.readByte(srcMem, srcOff);
            access.writeByte(dstMem, dstOff, ConstantFolder.evalByte(v, op));
        } else if (dtype == DataType.I16) {
            short v = access.readShort(srcMem, srcOff);
            access.writeShort(dstMem, dstOff, ConstantFolder.evalShort(v, op));
        } else if (dtype == DataType.I32) {
            int v = access.readInt(srcMem, srcOff);
            access.writeInt(dstMem, dstOff, ConstantFolder.evalInt(v, op));
        } else if (dtype == DataType.I64) {
            long v = access.readLong(srcMem, srcOff);
            access.writeLong(dstMem, dstOff, ConstantFolder.evalLong(v, op));
        } else if (dtype == DataType.FP32) {
            float v = access.readFloat(srcMem, srcOff);
            access.writeFloat(dstMem, dstOff, ConstantFolder.evalFloat(v, op));
        } else if (dtype == DataType.FP64) {
            double v = access.readDouble(srcMem, srcOff);
            access.writeDouble(dstMem, dstOff, ConstantFolder.evalDouble(v, op));
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    private MemoryAccess<Object> requireMemoryAccess() {
        MemoryAccess<?> access = context.memoryAccess();
        if (access == null) {
            throw new IllegalStateException("Eager ops require MemoryAccess");
        }
        @SuppressWarnings("unchecked")
        MemoryAccess<Object> cast = (MemoryAccess<Object>) access;
        return cast;
    }

    private static void copyBetweenContexts(
            MemoryContext<?> sourceContext,
            MemoryView<?> sourceView,
            MemoryContext<?> targetContext,
            MemoryView<?> targetView) {
        @SuppressWarnings("unchecked")
        MemoryContext<Object> source = (MemoryContext<Object>) sourceContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> src = (MemoryView<Object>) sourceView;
        @SuppressWarnings("unchecked")
        MemoryContext<Object> target = (MemoryContext<Object>) targetContext;
        @SuppressWarnings("unchecked")
        MemoryView<Object> dst = (MemoryView<Object>) targetView;
        MemoryContext.copy(source, src, target, dst);
    }

    private static OutputBufferSpec computeOutputSpec(Layout layout, DataType dataType) {
        long[] shape = layout.shape().toArray();
        long[] strideBytes = layout.stride().scale(dataType.byteSize()).toArray();
        long minOffset = 0;
        long maxOffset = 0;
        for (int i = 0; i < shape.length; i++) {
            long dim = shape[i];
            if (dim <= 1) {
                continue;
            }
            long span = (dim - 1) * strideBytes[i];
            if (strideBytes[i] >= 0) {
                maxOffset += span;
            } else {
                minOffset += span;
            }
        }
        long byteOffset = -minOffset;
        long byteSize = maxOffset - minOffset + dataType.byteSize();
        return new OutputBufferSpec(byteOffset, byteSize);
    }

    private record OutputBufferSpec(long byteOffset, long byteSize) {}

    private Tensor binaryOp(Tensor a, Tensor b, BinaryOp op) {
        MemoryView<?> viewA = a.materialize();
        MemoryView<?> viewB = b.materialize();
        DataType dtype = viewA.dataType();
        if (dtype != viewB.dataType()) {
            throw new IllegalArgumentException(
                    "Type mismatch: " + dtype + " vs " + viewB.dataType());
        }
        Shape resultShape = viewA.shape();
        if (!viewA.shape().equals(viewB.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + viewA.shape() + " vs " + viewB.shape());
        }
        MemoryAccess<Object> access = requireMemoryAccess();
        MemoryView<?> output = allocate(dtype, resultShape);
        long size = resultShape.size();
        @SuppressWarnings("unchecked")
        Memory<Object> memA = (Memory<Object>) viewA.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memB = (Memory<Object>) viewB.memory();
        @SuppressWarnings("unchecked")
        Memory<Object> memOut = (Memory<Object>) output.memory();
        for (long i = 0; i < size; i++) {
            long offA = Indexing.linearToOffset(viewA, i);
            long offB = Indexing.linearToOffset(viewB, i);
            long offOut = Indexing.linearToOffset(output, i);
            applyBinaryOp(access, memA, offA, memB, offB, memOut, offOut, dtype, op);
        }
        return Tensor.of(output);
    }

    private void applyBinaryOp(
            MemoryAccess<Object> access,
            Memory<Object> memA,
            long offA,
            Memory<Object> memB,
            long offB,
            Memory<Object> memOut,
            long offOut,
            DataType dtype,
            BinaryOp op) {
        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte a = access.readByte(memA, offA);
            byte b = access.readByte(memB, offB);
            access.writeByte(memOut, offOut, ConstantFolder.evalByte(a, b, op));
        } else if (dtype == DataType.I16) {
            short a = access.readShort(memA, offA);
            short b = access.readShort(memB, offB);
            access.writeShort(memOut, offOut, ConstantFolder.evalShort(a, b, op));
        } else if (dtype == DataType.I32) {
            int a = access.readInt(memA, offA);
            int b = access.readInt(memB, offB);
            access.writeInt(memOut, offOut, ConstantFolder.evalInt(a, b, op));
        } else if (dtype == DataType.I64) {
            long a = access.readLong(memA, offA);
            long b = access.readLong(memB, offB);
            access.writeLong(memOut, offOut, ConstantFolder.evalLong(a, b, op));
        } else if (dtype == DataType.FP32) {
            float a = access.readFloat(memA, offA);
            float b = access.readFloat(memB, offB);
            access.writeFloat(memOut, offOut, ConstantFolder.evalFloat(a, b, op));
        } else if (dtype == DataType.FP64) {
            double a = access.readDouble(memA, offA);
            double b = access.readDouble(memB, offB);
            access.writeDouble(memOut, offOut, ConstantFolder.evalDouble(a, b, op));
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + dtype);
        }
    }

    private Tensor scalarOp(Tensor a, Number scalar, BinaryOp op) {
        throw new UnsupportedOperationException("Generic scalar op dispatch not yet implemented");
    }

    private Tensor reductionOp(Tensor x, ReductionOp op, int axis, boolean keepDims) {
        throw new UnsupportedOperationException(
                "Generic reduction op dispatch not yet implemented");
    }
}

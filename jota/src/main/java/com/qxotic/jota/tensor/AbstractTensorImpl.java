package com.qxotic.jota.tensor;

import com.qxotic.jota.*;
import com.qxotic.jota.impl.ViewTransforms;
import com.qxotic.jota.memory.MemoryView;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.stream.IntStream;

abstract class AbstractTensorImpl implements Tensor {

    public abstract DataType dataType();

    public abstract Layout layout();

    public Shape shape() {
        return layout().shape();
    }

    public Stride stride() {
        return layout().stride();
    }

    public abstract Device device();

    public long size() {
        return shape().size();
    }

    public boolean isScalar() {
        return shape().isScalar();
    }

    final boolean isScalarBroadcastInternal() {
        Optional<MemoryView<?>> materialized = tryGetMaterializedInternal();
        if (materialized.isPresent()) {
            MemoryView<?> view = materialized.get();
            return view.isBroadcasted()
                    && Arrays.stream(view.stride().toArray()).allMatch(s -> s == 0L);
        }
        return computationInternal()
                .filter(ConstantComputation.class::isInstance)
                .map(
                        computation ->
                                Arrays.stream(layout().stride().toArray()).allMatch(s -> s == 0L))
                .orElse(false);
    }

    final OptionalLong scalarConstantBitsInternal() {
        Optional<ConstantComputation> constant =
                computationInternal()
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast);
        if (constant.isPresent()) {
            return OptionalLong.of(constant.get().rawBits());
        }
        return OptionalLong.empty();
    }

    abstract boolean isMaterializedInternal();

    abstract boolean isLazyInternal();

    public abstract MemoryView<?> materialize();

    abstract Optional<MemoryView<?>> tryGetMaterializedInternal();

    abstract Optional<LazyComputation> computationInternal();

    // region Elementwise Ops
    // region Binary Ops

    public Tensor add(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.ADD,
                (left, right) -> TensorSupport.irOps().add(left, right),
                Tensor::add);
    }

    public Tensor add(int scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::add);
    }

    public Tensor add(long scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::add);
    }

    public Tensor add(float scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::add);
    }

    public Tensor add(double scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::add);
    }

    public Tensor subtract(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.SUBTRACT,
                (left, right) -> TensorSupport.irOps().subtract(left, right),
                Tensor::subtract);
    }

    public Tensor subtract(int scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::subtract);
    }

    public Tensor subtract(long scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::subtract);
    }

    public Tensor subtract(float scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::subtract);
    }

    public Tensor subtract(double scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::subtract);
    }

    public Tensor multiply(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.MULTIPLY,
                (left, right) -> TensorSupport.irOps().multiply(left, right),
                Tensor::multiply);
    }

    public Tensor multiply(int scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::multiply);
    }

    public Tensor multiply(long scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::multiply);
    }

    public Tensor multiply(float scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::multiply);
    }

    public Tensor multiply(double scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::multiply);
    }

    public Tensor divide(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.DIVIDE,
                (left, right) -> TensorSupport.irOps().divide(left, right),
                Tensor::divide);
    }

    public Tensor divide(int scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::divide);
    }

    public Tensor divide(long scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::divide);
    }

    public Tensor divide(float scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::divide);
    }

    public Tensor divide(double scalar) {
        return TensorSupport.dispatchScalarBinaryOp(this, scalar, Tensor::divide);
    }

    public Tensor min(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.MIN,
                (left, right) -> TensorSupport.irOps().min(left, right),
                Tensor::min);
    }

    public Tensor min() {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().min(this);
        }
        return Tracer.trace(this, Tensor::min);
    }

    public Tensor min(int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().min(this, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.min(_axis, _axes));
    }

    public Tensor min(boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().min(this, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.min(keepDims, _axis, _axes));
    }

    public Tensor max(Tensor other) {
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.MAX,
                (left, right) -> TensorSupport.irOps().max(left, right),
                Tensor::max);
    }

    public Tensor max() {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().max(this);
        }
        return Tracer.trace(this, Tensor::max);
    }

    public Tensor max(int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().max(this, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.max(_axis, _axes));
    }

    public Tensor max(boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().max(this, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.max(keepDims, _axis, _axes));
    }

    // endregion Binary Ops
    // region Reduction / Linear Algebra Ops

    public Tensor argmax() {
        TensorSupport.requireNumericNonBool(dataType(), "argmax");
        if (shape().rank() == 0) {
            return Tensor.scalar(0L, DataType.I64);
        }
        Tensor flat = reshape(Shape.of(shape().size()));
        return flat.argmax(0, false);
    }

    public Tensor argmax(int _axis) {
        return argmax(_axis, false);
    }

    public Tensor argmax(int _axis, boolean keepDims) {
        TensorSupport.requireNumericNonBool(dataType(), "argmax");
        int axis = TensorSemantics.normalizeAxis(shape().rank(), _axis);
        long axisSize = shape().flatAt(axis);

        Tensor comparable = TensorSupport.argReduceComparableView(this);
        Tensor extrema = comparable.max(true, axis);
        Tensor mask = comparable.equal(extrema.broadcast(shape()));
        Tensor axisIndices = TensorSupport.axisIndexGrid(shape(), axis);
        Tensor sentinel = Tensor.full(axisSize, DataType.I64, shape());
        Tensor selected = mask.where(axisIndices, sentinel);
        return selected.min(keepDims, axis);
    }

    public Tensor argmin() {
        TensorSupport.requireNumericNonBool(dataType(), "argmin");
        if (shape().rank() == 0) {
            return Tensor.scalar(0L, DataType.I64);
        }
        Tensor flat = reshape(Shape.of(shape().size()));
        return flat.argmin(0, false);
    }

    public Tensor argmin(int _axis) {
        return argmin(_axis, false);
    }

    public Tensor argmin(int _axis, boolean keepDims) {
        TensorSupport.requireNumericNonBool(dataType(), "argmin");
        int axis = TensorSemantics.normalizeAxis(shape().rank(), _axis);
        long axisSize = shape().flatAt(axis);

        Tensor comparable = TensorSupport.argReduceComparableView(this);
        Tensor extrema = comparable.min(true, axis);
        Tensor mask = comparable.equal(extrema.broadcast(shape()));
        Tensor axisIndices = TensorSupport.axisIndexGrid(shape(), axis);
        Tensor sentinel = Tensor.full(axisSize, DataType.I64, shape());
        Tensor selected = mask.where(axisIndices, sentinel);
        return selected.min(keepDims, axis);
    }

    public Tensor any() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        if (shape().rank() == 0) {
            return this;
        }
        Tensor reduced = sum(DataType.I64);
        return reduced.greaterThan(Tensor.scalar(0L, DataType.I64));
    }

    public Tensor all() {
        if (dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("expected BOOL");
        }
        if (shape().rank() == 0) {
            return this;
        }
        Tensor reduced = sum(DataType.I64);
        return reduced.equal(Tensor.scalar(shape().size(), DataType.I64));
    }

    public Tensor matmul(Tensor other) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().matmul(this, other);
        }
        return Tracer.trace(this, other, (a, b) -> a.matmul(b));
    }

    public Tensor batchedMatmul(Tensor other) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().batchedMatmul(this, other);
        }
        return Tracer.trace(this, other, (a, b) -> a.batchedMatmul(b));
    }

    public Tensor dot(Tensor other, DataType accumulatorType) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(accumulatorType, "accumulatorType");

        TensorSupport.requireNumericNonBool(dataType(), "dot");
        TensorSupport.requireNumericNonBool(other.dataType(), "dot");
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
                TensorSupport.resolveReductionAccumulator(dataType(), accumulatorType, "dot");
        Tensor left = cast(accType);
        Tensor right = other.cast(accType);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().dot(left, right, accType);
        }
        return Tracer.trace(left, right, (a, b) -> a.dot(b, accType));
    }

    public Tensor dot(Tensor other) {
        Objects.requireNonNull(other, "other");
        if (!dataType().isFloatingPoint() || !other.dataType().isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "dot(other) is floating-point only; use dot(other, accumulatorType) for integral inputs");
        }
        return dot(other, dataType());
    }

    public Tensor to(Device device) {
        Objects.requireNonNull(device, "device");
        if (Tracer.isTracing()) {
            throw new UnsupportedOperationException(
                    "Tensor.to(Device) is a runtime transfer boundary and is not allowed inside tracing");
        }
        if (device.equals(device())) {
            return this;
        }
        return TensorSupport.transferToDevice(this, device);
    }

    // endregion Reduction / Linear Algebra Ops
    // region View / Shape Ops

    public Tensor contiguous() {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().contiguous(this);
        }
        return Tracer.trace(this, Tensor::contiguous);
    }

    public Tensor view(Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(layout(), newShape);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor view(long... dims) {
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

    public Tensor unsqueeze(int axis_) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.unsqueeze(layout(), axis_);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor squeeze(int _axis) {
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

    public Tensor squeezeAll() {
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

    public Tensor broadcast(Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.broadcast(layout(), targetShape);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor expand(Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.expand(layout(), targetShape);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor transpose(int _axis0, int _axis1) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.transpose(layout(), _axis0, _axis1);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor permute(int... permutationIndices) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.permute(layout(), permutationIndices);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor slice(int _axis, long start, long end) {
        return slice(_axis, start, end, 1);
    }

    public Tensor slice(int _axis, long start, long end, long indexStride) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.slice(layout(), dataType(), _axis, start, end, indexStride);
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().viewTransform(this, spec);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().viewTransform(t, spec));
    }

    public Tensor repeat(long... repeats) {
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
        return repeatedFlat.view(TensorSupport.resolveRepeatedShape(currentShape, repeats, out));
    }

    public Tensor repeatInterleave(long repeats, int _axis) {
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

    public Tensor reshape(Shape newShape) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().reshape(this, newShape);
        }
        return Tracer.trace(this, t -> TensorSupport.irOps().reshape(t, newShape));
    }

    // endregion View / Shape Ops
    // region Bitwise / Logical / Comparison Ops

    public Tensor bitwiseNot() {
        TensorSupport.requireIntegral(dataType(), "bitwiseNot");
        return TensorSupport.dispatchFoldedUnaryOp(
                this,
                UnaryOp.BITWISE_NOT,
                t -> TensorSupport.irOps().bitwiseNot(t),
                Tensor::bitwiseNot);
    }

    public Tensor bitwiseAnd(Tensor other) {
        TensorSupport.requireSameIntegralType(dataType(), other.dataType(), "bitwiseAnd");
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.BITWISE_AND,
                (left, right) -> TensorSupport.irOps().bitwiseAnd(left, right),
                Tensor::bitwiseAnd);
    }

    public Tensor bitwiseOr(Tensor other) {
        TensorSupport.requireSameIntegralType(dataType(), other.dataType(), "bitwiseOr");
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.BITWISE_OR,
                (left, right) -> TensorSupport.irOps().bitwiseOr(left, right),
                Tensor::bitwiseOr);
    }

    public Tensor bitwiseXor(Tensor other) {
        TensorSupport.requireSameIntegralType(dataType(), other.dataType(), "bitwiseXor");
        return TensorSupport.dispatchFoldedBinaryOp(
                this,
                other,
                BinaryOp.BITWISE_XOR,
                (left, right) -> TensorSupport.irOps().bitwiseXor(left, right),
                Tensor::bitwiseXor);
    }

    public Tensor leftShift(Tensor other) {
        return TensorSupport.dispatchShiftBinaryOp(
                this,
                other,
                "leftShift",
                BinaryOp.LEFT_SHIFT,
                (left, right) -> TensorSupport.irOps().leftShift(left, right),
                Tensor::leftShift);
    }

    public Tensor rightShift(Tensor other) {
        return TensorSupport.dispatchShiftBinaryOp(
                this,
                other,
                "rightShift",
                BinaryOp.RIGHT_SHIFT,
                (left, right) -> TensorSupport.irOps().rightShift(left, right),
                Tensor::rightShift);
    }

    public Tensor rightShiftUnsigned(Tensor other) {
        return TensorSupport.dispatchShiftBinaryOp(
                this,
                other,
                "rightShiftUnsigned",
                BinaryOp.RIGHT_SHIFT_UNSIGNED,
                (left, right) -> TensorSupport.irOps().rightShiftUnsigned(left, right),
                Tensor::rightShiftUnsigned);
    }

    public Tensor logicalNot() {
        TensorSupport.requireBool(dataType(), "logicalNot");
        return TensorSupport.dispatchFoldedUnaryOp(
                this,
                UnaryOp.LOGICAL_NOT,
                t -> TensorSupport.irOps().logicalNot(t),
                Tensor::logicalNot);
    }

    public Tensor logicalAnd(Tensor other) {
        TensorSupport.requireBooleanPair(dataType(), other.dataType(), "logicalAnd");
        return TensorSupport.dispatchBinaryOp(
                this,
                other,
                (left, right) -> TensorSupport.irOps().logicalAnd(left, right),
                Tensor::logicalAnd);
    }

    public Tensor logicalOr(Tensor other) {
        TensorSupport.requireBooleanPair(dataType(), other.dataType(), "logicalOr");
        return TensorSupport.dispatchBinaryOp(
                this,
                other,
                (left, right) -> TensorSupport.irOps().logicalOr(left, right),
                Tensor::logicalOr);
    }

    public Tensor logicalXor(Tensor other) {
        TensorSupport.requireBooleanPair(dataType(), other.dataType(), "logicalXor");
        return TensorSupport.dispatchBinaryOp(
                this,
                other,
                (left, right) -> TensorSupport.irOps().logicalXor(left, right),
                Tensor::logicalXor);
    }

    public Tensor equal(Tensor other) {
        return TensorSupport.dispatchFoldedCompareOp(
                this,
                other,
                BinaryOp.EQUAL,
                (left, right) -> TensorSupport.irOps().equal(left, right),
                Tensor::equal);
    }

    public Tensor lessThan(Tensor other) {
        return TensorSupport.dispatchFoldedCompareOp(
                this,
                other,
                BinaryOp.LESS_THAN,
                (left, right) -> TensorSupport.irOps().lessThan(left, right),
                Tensor::lessThan);
    }

    // endregion Bitwise / Logical / Comparison Ops
    // region Selection / Reduction Ops

    public Tensor where(Tensor trueValue, Tensor falseValue) {
        TensorSupport.requireBool(dataType(), "where condition");
        if (trueValue.dataType() != falseValue.dataType()) {
            throw new IllegalArgumentException(
                    "where requires true and false values to have the same type, got "
                            + trueValue.dataType()
                            + " and "
                            + falseValue.dataType());
        }
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().where(this, trueValue, falseValue);
        }
        return Tracer.trace(this, trueValue, falseValue, (c, t, f) -> c.where(t, f));
    }

    public Tensor sum(DataType accumulatorType) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().sum(this, accumulatorType);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType));
    }

    public Tensor sum(DataType accumulatorType, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().sum(this, accumulatorType, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType, _axis, _axes));
    }

    public Tensor sum(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().sum(this, accumulatorType, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.sum(accumulatorType, keepDims, _axis, _axes));
    }

    public Tensor product(DataType accumulatorType) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().product(this, accumulatorType);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType));
    }

    public Tensor product(DataType accumulatorType, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().product(this, accumulatorType, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType, _axis, _axes));
    }

    public Tensor product(DataType accumulatorType, boolean keepDims, int _axis, int... _axes) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().product(this, accumulatorType, keepDims, _axis, _axes);
        }
        return Tracer.trace(this, t -> t.product(accumulatorType, keepDims, _axis, _axes));
    }

    public Tensor mean() {
        TensorSupport.requireFloatingPoint(dataType(), "mean");
        int rank = shape().rank();
        if (rank == 0) {
            return this;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return mean(false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    public Tensor mean(int _axis, int... _axes) {
        return mean(false, _axis, _axes);
    }

    public Tensor mean(boolean keepDims, int _axis, int... _axes) {
        TensorSupport.requireFloatingPoint(dataType(), "mean");
        int[] axes = TensorSemantics.normalizeReductionAxes(shape().rank(), _axis, _axes);
        long count = 1L;
        for (int axis : axes) {
            count *= shape().flatAt(axis);
        }
        Tensor reduced = sum(dataType(), keepDims, _axis, _axes);
        return reduced.divide(Tensor.scalar((double) count, dataType()));
    }

    public Tensor gather(Tensor indices, int _axis) {
        if (Tracer.isTracing()) {
            return TensorSupport.irOps().gather(this, indices, _axis);
        }
        return Tracer.trace(this, indices, (input, idx) -> input.gather(idx, _axis));
    }

    public Tensor embeddingLookup(Tensor indices) {
        return gather(indices, 0);
    }

    // endregion Selection / Reduction Ops
    // region Unary Ops

    public Tensor cast(DataType targetType) {
        return TensorSupport.dispatchFoldedCastOp(
                this,
                targetType,
                (t, dt) -> TensorSupport.irOps().cast(t, dt),
                (t, dt) -> t.cast(dt));
    }

    public Tensor negate() {
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.NEGATE, t -> TensorSupport.irOps().negate(t), Tensor::negate);
    }

    public Tensor abs() {
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.ABS, t -> TensorSupport.irOps().abs(t), Tensor::abs);
    }

    public Tensor exp() {
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.EXP, t -> TensorSupport.irOps().exp(t), Tensor::exp);
    }

    public Tensor log() {
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.LOG, t -> TensorSupport.irOps().log(t), Tensor::log);
    }

    public Tensor sqrt() {
        TensorSupport.requireFloatingPoint(dataType(), "sqrt");
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.SQRT, t -> TensorSupport.irOps().sqrt(t), Tensor::sqrt);
    }

    public Tensor sin() {
        TensorSupport.requireFloatingPoint(dataType(), "sin");
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.SIN, t -> TensorSupport.irOps().sin(t), Tensor::sin);
    }

    public Tensor cos() {
        TensorSupport.requireFloatingPoint(dataType(), "cos");
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.COS, t -> TensorSupport.irOps().cos(t), Tensor::cos);
    }

    public Tensor tanh() {
        TensorSupport.requireFloatingPoint(dataType(), "tanh");
        return TensorSupport.dispatchFoldedUnaryOp(
                this, UnaryOp.TANH, t -> TensorSupport.irOps().tanh(t), Tensor::tanh);
    }

    public Tensor relu() {
        TensorSupport.requireFloatingPoint(dataType(), "relu");
        return max(Tensor.full(0f, dataType(), shape()));
    }

    public Tensor sigmoid() {
        TensorSupport.requireFloatingPoint(dataType(), "sigmoid");
        return negate().exp().add(Tensor.scalar(1, dataType())).reciprocal();
    }

    public Tensor silu() {
        TensorSupport.requireFloatingPoint(dataType(), "silu");
        return multiply(sigmoid()); // x * sigmoid(x)
    }

    public Tensor gelu() {
        TensorSupport.requireFloatingPoint(dataType(), "gelu");
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

    public Tensor reciprocal() {
        TensorSupport.requireFloatingPoint(dataType(), "reciprocal");
        return TensorSupport.dispatchFoldedUnaryOp(
                this,
                UnaryOp.RECIPROCAL,
                t -> TensorSupport.irOps().reciprocal(t),
                Tensor::reciprocal);
    }

    public Tensor reciprocal(DataType dataType) {
        if (!dataType.isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "reciprocal target type must be floating-point, got " + dataType);
        }
        return this.cast(dataType).reciprocal();
    }

    // endregion Unary Ops
    // endregion Elementwise Ops
}

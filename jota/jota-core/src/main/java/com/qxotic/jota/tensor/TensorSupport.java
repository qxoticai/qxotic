package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.BinaryOp;
import com.qxotic.jota.runtime.UnaryOp;
import java.util.Arrays;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;

final class TensorSupport {

    private TensorSupport() {}

    static TensorOps irOps() {
        return Tracer.withRequiredIROps(ops -> ops);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    static Tensor transferToDevice(Tensor source, Device targetDevice) {
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

    static Tensor broadcastLeftScalar(Tensor left, Tensor right) {
        if (!left.isScalar() || right.isScalar()) {
            return left;
        }
        Optional<Tensor> constant =
                InternalTensorAccess.computation(left)
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast)
                        .map(c -> Tensor.full(c.value(), c.dataType(), right.shape()));
        if (constant.isPresent()) {
            return constant.get();
        }
        return left.broadcast(right.shape());
    }

    static Tensor broadcastRightScalar(Tensor left, Tensor right) {
        if (!right.isScalar() || left.isScalar()) {
            return right;
        }
        Optional<Tensor> constant =
                InternalTensorAccess.computation(right)
                        .filter(ConstantComputation.class::isInstance)
                        .map(ConstantComputation.class::cast)
                        .map(c -> Tensor.full(c.value(), c.dataType(), left.shape()));
        if (constant.isPresent()) {
            return constant.get();
        }
        return right.broadcast(left.shape());
    }

    static Tensor axisIndexGrid(Shape shape, int axis) {
        long[] indexShapeDims = new long[shape.rank()];
        Arrays.fill(indexShapeDims, 1L);
        indexShapeDims[axis] = shape.flatAt(axis);
        Tensor axisIndices =
                Tensor.iota(shape.flatAt(axis), DataType.I64).view(Shape.flat(indexShapeDims));
        return axisIndices.broadcast(shape);
    }

    static Tensor argReduceComparableView(Tensor input) {
        DataType dataType = input.dataType();
        if (dataType == DataType.FP16 || dataType == DataType.BF16) {
            return input.cast(DataType.FP32);
        }
        return input;
    }

    static void requireSameIntegralType(DataType left, DataType right, String opName) {
        TensorTypeSemantics.requireSameIntegralType(left, right, opName);
    }

    static void requireBooleanPair(DataType left, DataType right, String opName) {
        TensorTypeSemantics.requireBooleanPair(left, right, opName);
    }

    static void requireIntegral(DataType dataType, String opName) {
        TensorTypeSemantics.requireIntegral(dataType, opName);
    }

    static void requireFloatingPoint(DataType dataType, String opName) {
        TensorTypeSemantics.requireFloatingPoint(dataType, opName);
    }

    static void requireNumericNonBool(DataType dataType, String opName) {
        TensorTypeSemantics.requireNumericNonBool(dataType, opName);
    }

    static void requireBool(DataType dataType, String opName) {
        TensorTypeSemantics.requireBool(dataType, opName);
    }

    static DataType resolveReductionAccumulator(
            DataType inputType, DataType accumulatorType, String opName) {
        return TensorTypeSemantics.resolveReductionAccumulator(inputType, accumulatorType, opName);
    }

    static Tensor dispatchBinaryOp(
            Tensor self,
            Tensor other,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        Tensor left = broadcastLeftScalar(self, other);
        Tensor right = broadcastRightScalar(self, other);
        if (Tracer.isTracing()) {
            return tracedOp.apply(left, right);
        }
        return Tracer.trace(left, right, eagerOp);
    }

    static Tensor dispatchScalarBinaryOp(
            Tensor self, Number scalar, BiFunction<Tensor, Tensor, Tensor> binaryOp) {
        Tensor scalarTensor;
        if (scalar instanceof Integer value) {
            scalarTensor = Tensor.broadcasted(value, self.shape());
        } else if (scalar instanceof Long value) {
            scalarTensor = Tensor.broadcasted(value, self.shape());
        } else if (scalar instanceof Float value) {
            scalarTensor = Tensor.broadcasted(value, self.shape());
        } else if (scalar instanceof Double value) {
            scalarTensor = Tensor.broadcasted(value, self.shape());
        } else {
            throw new IllegalArgumentException("Unsupported scalar type: " + scalar.getClass());
        }
        return binaryOp.apply(self, scalarTensor);
    }

    static Tensor dispatchFoldedBinaryOp(
            Tensor self,
            Tensor other,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        return ConstantFolder.tryFoldBinaryOp(self, other, foldOp)
                .orElseGet(() -> dispatchBinaryOp(self, other, tracedOp, eagerOp));
    }

    static Tensor dispatchFoldedCompareOp(
            Tensor self,
            Tensor other,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        return ConstantFolder.tryFoldCompareOp(self, other, foldOp)
                .orElseGet(() -> dispatchBinaryOp(self, other, tracedOp, eagerOp));
    }

    static Tensor dispatchFoldedUnaryOp(
            Tensor self,
            UnaryOp foldOp,
            Function<Tensor, Tensor> tracedOp,
            Function<Tensor, Tensor> eagerOp) {
        return ConstantFolder.tryFoldUnaryOp(self, foldOp)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return tracedOp.apply(self);
                            }
                            return Tracer.trace(self, eagerOp::apply);
                        });
    }

    static Tensor dispatchFoldedCastOp(
            Tensor self,
            DataType targetType,
            BiFunction<Tensor, DataType, Tensor> tracedOp,
            BiFunction<Tensor, DataType, Tensor> eagerOp) {
        if (self.dataType() == targetType) {
            return self;
        }
        return ConstantFolder.tryFoldCast(self, targetType)
                .orElseGet(
                        () -> {
                            if (Tracer.isTracing()) {
                                return tracedOp.apply(self, targetType);
                            }
                            return Tracer.trace(self, t -> eagerOp.apply(t, targetType));
                        });
    }

    static Tensor dispatchShiftBinaryOp(
            Tensor self,
            Tensor other,
            String opName,
            BinaryOp foldOp,
            BiFunction<Tensor, Tensor, Tensor> tracedOp,
            BiFunction<Tensor, Tensor, Tensor> eagerOp) {
        DataType shiftType = other.dataType();
        TensorTypeSemantics.requireShiftOperandTypes(self.dataType(), shiftType, opName);
        Tensor normalizedOther = shiftType == DataType.I32 ? other : other.cast(DataType.I32);
        if (self.dataType() == normalizedOther.dataType()) {
            return dispatchFoldedBinaryOp(self, normalizedOther, foldOp, tracedOp, eagerOp);
        }
        return dispatchBinaryOp(self, normalizedOther, tracedOp, eagerOp);
    }

    static Shape resolveRepeatedShape(Shape shape, long[] repeats, long[] outModeSizes) {
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
}

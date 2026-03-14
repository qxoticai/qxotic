package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.ir.tir.IotaConstant;
import com.qxotic.jota.ir.tir.ScalarConstant;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.random.RandomKeys;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class TensorFactory {

    private TensorFactory() {}

    private static Device defaultDevice() {
        return Device.defaultDevice();
    }

    private static MemoryDomain<?> defaultMemoryDomain() {
        return Environment.current().runtimeFor(defaultDevice()).memoryDomain();
    }

    private static Tensor onDefaultDevice(Number value, DataType dataType, Shape shape) {
        return broadcasted(value, dataType, shape, defaultDevice());
    }

    private static Tensor viewWithShape(Tensor tensor, Shape shape) {
        Objects.requireNonNull(shape, "shape");
        return tensor.view(shape);
    }

    @FunctionalInterface
    private interface ArrayCopier {
        MemoryView<?> copy(MemoryDomain<?> memoryDomain, Shape shape);
    }

    private static Tensor fromArray(int length, Shape shape, ArrayCopier copier) {
        requireArraySize(length, shape);
        MemoryView<?> view = copier.copy(defaultMemoryDomain(), shape);
        return of(view);
    }

    // region Tensor Creation
    // region Core Construction
    static Tensor of(MemoryView<?> view) {
        return new MaterializedTensorImpl(view);
    }

    static Tensor lazy(LazyComputation computation, DataType dtype, Layout layout, Device device) {
        return new LazyTensorImpl(computation, dtype, layout, device);
    }

    static Tensor iota(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got: " + n);
        }
        Shape shape = Shape.flat(n);
        Device device = defaultDevice();
        if (Tracer.isTracing()) {
            return new IRTensorImpl(new IotaConstant(n, DataType.I64, shape), device);
        }
        Layout layout = Layout.rowMajor(shape);
        RangeComputation computation = new RangeComputation(n, device);
        return lazy(computation, DataType.I64, layout, device);
    }

    static Tensor iota(long n, DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got: " + n);
        }
        if (dataType == DataType.BOOL || !(dataType.isIntegral() || dataType.isFloatingPoint())) {
            throw new IllegalArgumentException("Unsupported data type for iota: " + dataType);
        }
        if (Tracer.isTracing()) {
            return new IRTensorImpl(new IotaConstant(n, dataType, Shape.flat(n)), defaultDevice());
        }
        return iota(n).cast(dataType);
    }

    // endregion Core Construction
    // region Random Creation

    static Tensor zeros(Shape shape) {
        return zeros(DataType.defaultFloat(), shape);
    }

    static Tensor zeros(DataType dtype, Shape shape) {
        return onDefaultDevice(0, dtype, shape);
    }

    static Tensor ones(Shape shape) {
        return ones(DataType.defaultFloat(), shape);
    }

    static Tensor ones(DataType dtype, Shape shape) {
        return onDefaultDevice(1, dtype, shape);
    }

    static RandomKey randomKey(long seed) {
        return RandomKeys.key(seed);
    }

    static Tensor rand(RandomKey randomKey, long size, DataType dataType) {
        return randomUnitInterval(randomKey, size, dataType, "rand");
    }

    static Tensor rand(RandomKey randomKey, Shape shape, DataType dataType) {
        return uniform(randomKey, shape, 0.0, 1.0, dataType);
    }

    static Tensor randn(RandomKey randomKey, long size, DataType dataType) {
        return randomStandardNormal(randomKey, size, dataType, "randn");
    }

    static Tensor randn(RandomKey randomKey, Shape shape, DataType dataType) {
        return normal(randomKey, shape, 0.0, 1.0, dataType);
    }

    static Tensor randInt(
            RandomKey randomKey,
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType) {
        return uniformInt(randomKey, startInclusive, endExclusive, size, dataType);
    }

    static Tensor randInt(
            RandomKey randomKey,
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType) {
        return uniformInt(randomKey, startInclusive, endExclusive, shape, dataType);
    }

    static Tensor uniform(
            RandomKey randomKey,
            long size,
            double startInclusive,
            double endExclusive,
            DataType dataType) {
        ensureValidRandomFloatRange(startInclusive, endExclusive, "uniform");
        Tensor unit = rand(randomKey, size, dataType);
        if (dataType == DataType.FP32) {
            float span = (float) (endExclusive - startInclusive);
            float start = (float) startInclusive;
            return unit.multiply(span).add(start);
        }
        double span = endExclusive - startInclusive;
        return unit.multiply(span).add(startInclusive);
    }

    static Tensor uniform(
            RandomKey randomKey,
            Shape shape,
            double startInclusive,
            double endExclusive,
            DataType dataType) {
        return viewWithShape(
                uniform(randomKey, shape.size(), startInclusive, endExclusive, dataType), shape);
    }

    static Tensor normal(
            RandomKey randomKey, long size, double mean, double std, DataType dataType) {
        ensureValidNormalParams(mean, std, "normal");
        Tensor standard = randn(randomKey, size, dataType);
        if (dataType == DataType.FP32) {
            return standard.multiply((float) std).add((float) mean);
        }
        return standard.multiply(std).add(mean);
    }

    static Tensor normal(
            RandomKey randomKey, Shape shape, double mean, double std, DataType dataType) {
        return viewWithShape(normal(randomKey, shape.size(), mean, std, dataType), shape);
    }

    static Tensor uniformInt(
            RandomKey randomKey,
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "uniformInt");
        ensureValidRandIntRange(startInclusive, endExclusive, dataType, "uniformInt");

        long range = Math.subtractExact(endExclusive, startInclusive);

        Tensor uniform = randomUnitInterval(randomKey, size, DataType.FP64, "uniformInt");
        Tensor offset = uniform.multiply((double) range).cast(DataType.I64);
        Tensor shifted = offset.add(startInclusive);
        return dataType == DataType.I64 ? shifted : shifted.cast(dataType);
    }

    static Tensor uniformInt(
            RandomKey randomKey,
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType) {
        return viewWithShape(
                uniformInt(randomKey, startInclusive, endExclusive, shape.size(), dataType), shape);
    }

    static Tensor normalInt(
            RandomKey randomKey, long size, double mean, double std, DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "normalInt");
        ensureValidNormalParams(mean, std, "normalInt");

        Tensor fp = normal(randomKey, size, mean, std, DataType.FP64);
        Tensor rounded =
                fp.greaterThanOrEqual(Tensor.scalar(0.0, DataType.FP64))
                        .where(fp.add(0.5), fp.subtract(0.5))
                        .cast(DataType.I64);
        Tensor clamped =
                rounded.max(Tensor.scalar(randomIntMin(dataType), DataType.I64))
                        .min(Tensor.scalar(randomIntMax(dataType), DataType.I64));
        return dataType == DataType.I64 ? clamped : clamped.cast(dataType);
    }

    static Tensor normalInt(
            RandomKey randomKey, Shape shape, double mean, double std, DataType dataType) {
        return viewWithShape(normalInt(randomKey, shape.size(), mean, std, dataType), shape);
    }

    // endregion Random Creation
    // region Constant / Fill Creation

    static Tensor full(float value, Shape shape) {
        return onDefaultDevice(value, DataType.FP32, shape);
    }

    static Tensor full(double value, Shape shape) {
        return onDefaultDevice(value, DataType.FP64, shape);
    }

    static Tensor full(long value, Shape shape) {
        return onDefaultDevice(value, DataType.I64, shape);
    }

    static Tensor full(int value, Shape shape) {
        return onDefaultDevice(value, DataType.I32, shape);
    }

    static Tensor full(Number value, DataType dtype, Shape shape) {
        return onDefaultDevice(value, dtype, shape);
    }

    static Tensor broadcasted(int value, Shape shape) {
        return onDefaultDevice(value, DataType.I32, shape);
    }

    static Tensor broadcasted(long value, Shape shape) {
        return onDefaultDevice(value, DataType.I64, shape);
    }

    static Tensor broadcasted(float value, Shape shape) {
        return onDefaultDevice(value, DataType.FP32, shape);
    }

    static Tensor broadcasted(double value, Shape shape) {
        return onDefaultDevice(value, DataType.FP64, shape);
    }

    static Tensor scalar(int value) {
        return onDefaultDevice(value, DataType.I32, Shape.scalar());
    }

    static Tensor scalar(float value) {
        return onDefaultDevice(value, DataType.FP32, Shape.scalar());
    }

    static Tensor scalar(double value) {
        return onDefaultDevice(value, DataType.FP64, Shape.scalar());
    }

    static Tensor scalar(long value) {
        return onDefaultDevice(value, DataType.I64, Shape.scalar());
    }

    static Tensor scalar(double value, DataType dtype) {
        return onDefaultDevice(value, dtype, Shape.scalar());
    }

    static Tensor scalar(long value, DataType dtype) {
        return onDefaultDevice(value, dtype, Shape.scalar());
    }

    // endregion Constant / Fill Creation
    // region Structural Static Ops

    static Tensor concat(int _axis, Tensor first, Tensor second, Tensor... rest) {
        return TensorStaticOps.concat(_axis, first, second, rest);
    }

    static Tensor stack(int _axis, Tensor first, Tensor second, Tensor... rest) {
        return TensorStaticOps.stack(_axis, first, second, rest);
    }

    static Tensor[] split(
            int _axis, Tensor input, long firstSize, long secondSize, long... restSizes) {
        return TensorStaticOps.split(_axis, input, firstSize, secondSize, restSizes);
    }

    // endregion Structural Static Ops
    // region Array-backed Creation

    static Tensor of(float[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(float[] data, Shape shape) {
        return fromArray(
                data.length, shape, (memoryDomain, s) -> copyFloatArray(memoryDomain, data, s));
    }

    static Tensor of(double[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(double[] data, Shape shape) {
        return fromArray(
                data.length, shape, (memoryDomain, s) -> copyDoubleArray(memoryDomain, data, s));
    }

    static Tensor of(int[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(int[] data, Shape shape) {
        return fromArray(
                data.length, shape, (memoryDomain, s) -> copyIntArray(memoryDomain, data, s));
    }

    static Tensor of(long[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(long[] data, Shape shape) {
        return fromArray(
                data.length, shape, (memoryDomain, s) -> copyLongArray(memoryDomain, data, s));
    }

    static Tensor of(boolean[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(boolean[] data, Shape shape) {
        return fromArray(
                data.length, shape, (memoryDomain, s) -> copyBooleanArray(memoryDomain, data, s));
    }

    // endregion Array-backed Creation
    // endregion Tensor Creation

    // Operation/runtime helpers moved to TensorSupport.

    // region Internal Private Helpers
    // region Constant / IR Helpers

    private static Tensor broadcasted(Number value, DataType dataType, Shape shape, Device device) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(device, "device");
        Layout layout = Layout.of(shape, Stride.zeros(shape));

        if (Tracer.isTracing()) {
            return createIRScalarConstant(value, dataType, shape, device);
        }

        ConstantComputation computation = ConstantComputation.of(value, dataType, shape, device);
        return lazy(computation, dataType, layout, device);
    }

    // endregion Constant / IR Helpers

    // region Random Validation / Sampling Helpers

    private static void ensureSupportedRandomFloat(DataType dtype, String opName) {
        if (dtype != DataType.FP32 && dtype != DataType.FP64) {
            throw new IllegalArgumentException(opName + " supports FP32/FP64 only, got: " + dtype);
        }
    }

    private static Tensor randomUnitInterval(
            RandomKey randomKey, long size, DataType dataType, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);
        Shape shape = shapeFromSize(size, opName);
        Layout layout = Layout.rowMajor(shape);
        Device device = defaultDevice();
        RandomComputation computation = new RandomComputation(shape, dataType, device, randomKey);
        return lazy(computation, dataType, layout, device);
    }

    private static Tensor randomStandardNormal(
            RandomKey randomKey, long size, DataType dataType, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);

        RandomKey k0 = randomKey.split(0L);
        RandomKey k1 = randomKey.split(1L);
        Tensor u1 = randomUnitInterval(k0, size, dataType, opName);
        Tensor u2 = randomUnitInterval(k1, size, dataType, opName);

        if (dataType == DataType.FP64) {
            Tensor safeU2 = u2.clip(1.0e-15, 1.0 - 1.0e-15);
            Tensor r = safeU2.multiply(-1.0).add(1.0).log().multiply(-2.0).sqrt();
            Tensor theta = u1.multiply(2.0 * Math.PI);
            return theta.cos().multiply(r);
        }

        Tensor safeU2 = u2.clip(1.0e-7, 1.0 - 1.0e-7);
        Tensor r = safeU2.multiply(-1.0f).add(1.0f).log().multiply(-2.0f).sqrt();
        Tensor theta = u1.multiply((float) (2.0 * Math.PI));
        return theta.cos().multiply(r);
    }

    private static void ensureSupportedRandomInt(DataType dtype, String opName) {
        if (dtype == DataType.BOOL || !dtype.isIntegral()) {
            throw new IllegalArgumentException(
                    opName + " supports I8/I16/I32/I64 only, got: " + dtype);
        }
    }

    private static void ensureValidRandomFloatRange(
            double startInclusive, double endExclusive, String opName) {
        if (!Double.isFinite(startInclusive) || !Double.isFinite(endExclusive)) {
            throw new IllegalArgumentException(opName + " requires finite bounds");
        }
        if (!(endExclusive > startInclusive)) {
            throw new IllegalArgumentException(
                    opName
                            + " requires endExclusive > startInclusive, got ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ")");
        }
    }

    private static void ensureValidNormalParams(double mean, double std, String opName) {
        if (!Double.isFinite(mean) || !Double.isFinite(std)) {
            throw new IllegalArgumentException(opName + " requires finite mean/std");
        }
        if (!(std > 0.0)) {
            throw new IllegalArgumentException(opName + " requires std > 0, got: " + std);
        }
    }

    private static void ensureValidRandIntRange(
            long startInclusive, long endExclusive, DataType dataType, String opName) {
        if (endExclusive <= startInclusive) {
            throw new IllegalArgumentException(
                    opName
                            + " requires endExclusive > startInclusive, got ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ")");
        }
        long min = randomIntMin(dataType);
        long max = randomIntMax(dataType);
        long upperInclusive = endExclusive - 1L;
        if (startInclusive < min || upperInclusive > max) {
            throw new IllegalArgumentException(
                    opName
                            + " range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") does not fit dtype "
                            + dataType);
        }
    }

    private static Shape shapeFromSize(long size, String opName) {
        if (size < 0) {
            throw new IllegalArgumentException(
                    opName + " requires non-negative size, got: " + size);
        }
        return Shape.flat(size);
    }

    private static void requireArraySize(int length, Shape shape) {
        Objects.requireNonNull(shape, "shape");
        if (length != shape.size()) {
            throw new IllegalArgumentException(
                    "array length " + length + " does not match shape size " + shape.size());
        }
    }

    private static long randomIntMin(DataType dataType) {
        if (dataType == DataType.I8) {
            return Byte.MIN_VALUE;
        }
        if (dataType == DataType.I16) {
            return Short.MIN_VALUE;
        }
        if (dataType == DataType.I32) {
            return Integer.MIN_VALUE;
        }
        if (dataType == DataType.I64) {
            return Long.MIN_VALUE;
        }
        throw new IllegalArgumentException("Unsupported random integer dtype: " + dataType);
    }

    private static long randomIntMax(DataType dataType) {
        if (dataType == DataType.I8) {
            return Byte.MAX_VALUE;
        }
        if (dataType == DataType.I16) {
            return Short.MAX_VALUE;
        }
        if (dataType == DataType.I32) {
            return Integer.MAX_VALUE;
        }
        if (dataType == DataType.I64) {
            return Long.MAX_VALUE;
        }
        throw new IllegalArgumentException("Unsupported random integer dtype: " + dataType);
    }

    // endregion Random Validation / Sampling Helpers

    // region IR Scalar Constant Helpers

    private static Tensor createIRScalarConstant(
            Number value, DataType dataType, Shape shape, Device device) {
        ScalarConstant scalar;
        long rawBits;
        if (dataType == DataType.FP32) {
            rawBits = Float.floatToIntBits(value.floatValue());
        } else if (dataType == DataType.FP64) {
            rawBits = Double.doubleToLongBits(value.doubleValue());
        } else if (dataType == DataType.BOOL || dataType.isIntegral()) {
            rawBits = value.longValue();
        } else if (dataType == DataType.FP16 || dataType == DataType.BF16) {
            rawBits = (long) Float.floatToIntBits(value.floatValue());
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
        scalar = ScalarConstant.broadcast(rawBits, dataType, shape);
        return new IRTensorImpl(scalar, device);
    }

    // endregion IR Scalar Constant Helpers

    // Structural helpers moved to TensorStaticOps.

    // region Array Copy Helpers

    private static <B> MemoryView<B> copyFloatArray(
            MemoryDomain<B> memoryDomain, float[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.FP32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Float.BYTES);
        return MemoryView.of(dst, DataType.FP32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyDoubleArray(
            MemoryDomain<B> memoryDomain, double[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.FP64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Double.BYTES);
        return MemoryView.of(dst, DataType.FP64, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyIntArray(
            MemoryDomain<B> memoryDomain, int[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.I32, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Integer.BYTES);
        return MemoryView.of(dst, DataType.I32, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyLongArray(
            MemoryDomain<B> memoryDomain, long[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.I64, data.length);
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Long.BYTES);
        return MemoryView.of(dst, DataType.I64, Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> copyBooleanArray(
            MemoryDomain<B> memoryDomain, boolean[] data, Shape shape) {
        Memory<B> dst = memoryDomain.memoryAllocator().allocateMemory(DataType.BOOL, data.length);
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = data[i] ? (byte) 1 : 0;
        }
        Memory<MemorySegment> src = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(bytes));
        MemoryOperations<MemorySegment> srcOps = DomainFactory.ofMemorySegment().memoryOperations();
        MemoryOperations.copy(
                srcOps,
                src,
                0,
                memoryDomain.memoryOperations(),
                dst,
                0,
                (long) data.length * Byte.BYTES);
        return MemoryView.of(dst, DataType.BOOL, Layout.rowMajor(shape));
    }

    // endregion Array Copy Helpers
    // endregion Internal Private Helpers
}

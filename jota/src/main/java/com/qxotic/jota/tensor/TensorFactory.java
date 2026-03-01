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
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

final class TensorFactory {

    private static final long DEFAULT_SEED = 0x5eed5eedL;

    private static final ThreadLocal<RandomState> RANDOM_STATE =
            ThreadLocal.withInitial(
                    () -> new RandomState(RandomKey.of(DEFAULT_SEED), new AtomicLong(0L)));

    private TensorFactory() {}

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
        if (Tracer.isTracing()) {
            return new IRTensorImpl(
                    new IotaConstant(n, DataType.I64, shape), Device.defaultDevice());
        }
        Layout layout = Layout.rowMajor(shape);
        RangeComputation computation = new RangeComputation(n, Device.defaultDevice());
        return lazy(computation, DataType.I64, layout, Device.defaultDevice());
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
            return new IRTensorImpl(
                    new IotaConstant(n, dataType, Shape.flat(n)), Device.defaultDevice());
        }
        return iota(n).cast(dataType);
    }

    // endregion Core Construction
    // region Random Creation

    static Tensor zeros(Shape shape) {
        return zeros(DataType.defaultFloat(), shape);
    }

    static Tensor zeros(DataType dtype, Shape shape) {
        return broadcasted(0, dtype, shape, Device.defaultDevice());
    }

    static Tensor ones(Shape shape) {
        return ones(DataType.defaultFloat(), shape);
    }

    static Tensor ones(DataType dtype, Shape shape) {
        return broadcasted(1, dtype, shape, Device.defaultDevice());
    }

    static void manualSeed(long seed) {
        RANDOM_STATE.set(new RandomState(RandomKey.of(seed), new AtomicLong(0L)));
    }

    static Tensor rand(long size, DataType dataType) {
        return rand(size, dataType, nextKey());
    }

    static Tensor rand(long size, DataType dataType, RandomKey randomKey) {
        return randomUnitInterval(size, dataType, randomKey, "rand");
    }

    static Tensor rand(Shape shape, DataType dataType) {
        return uniform(shape, 0.0, 1.0, dataType);
    }

    static Tensor rand(Shape shape, DataType dataType, RandomKey randomKey) {
        return uniform(shape, 0.0, 1.0, dataType, randomKey);
    }

    static Tensor randn(long size, DataType dataType) {
        return randn(size, dataType, nextKey());
    }

    static Tensor randn(long size, DataType dataType, RandomKey randomKey) {
        return randomStandardNormal(size, dataType, randomKey, "randn");
    }

    static Tensor randn(Shape shape, DataType dataType) {
        return normal(shape, 0.0, 1.0, dataType);
    }

    static Tensor randn(Shape shape, DataType dataType, RandomKey randomKey) {
        return normal(shape, 0.0, 1.0, dataType, randomKey);
    }

    static Tensor randInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return uniformInt(startInclusive, endExclusive, size, dataType);
    }

    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        return uniformInt(startInclusive, endExclusive, size, dataType, randomKey);
    }

    static Tensor randInt(long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        return uniformInt(startInclusive, endExclusive, shape, dataType);
    }

    static Tensor randInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        return uniformInt(startInclusive, endExclusive, shape, dataType, randomKey);
    }

    static Tensor uniform(
            long size, double startInclusive, double endExclusive, DataType dataType) {
        return uniform(size, startInclusive, endExclusive, dataType, nextKey());
    }

    static Tensor uniform(
            long size,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        ensureValidRandomFloatRange(startInclusive, endExclusive, "uniform");
        Tensor unit = rand(size, dataType, randomKey);
        if (dataType == DataType.FP32) {
            float span = (float) (endExclusive - startInclusive);
            float start = (float) startInclusive;
            return unit.multiply(span).add(start);
        }
        double span = endExclusive - startInclusive;
        return unit.multiply(span).add(startInclusive);
    }

    static Tensor uniform(
            Shape shape, double startInclusive, double endExclusive, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return uniform(shape, startInclusive, endExclusive, dataType, nextKey());
    }

    static Tensor uniform(
            Shape shape,
            double startInclusive,
            double endExclusive,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return uniform(shape.size(), startInclusive, endExclusive, dataType, randomKey).view(shape);
    }

    static Tensor normal(long size, double mean, double std, DataType dataType) {
        return normal(size, mean, std, dataType, nextKey());
    }

    static Tensor normal(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        ensureValidNormalParams(mean, std, "normal");
        Tensor standard = randn(size, dataType, randomKey);
        if (dataType == DataType.FP32) {
            return standard.multiply((float) std).add((float) mean);
        }
        return standard.multiply(std).add(mean);
    }

    static Tensor normal(Shape shape, double mean, double std, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return normal(shape, mean, std, dataType, nextKey());
    }

    static Tensor normal(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return normal(shape.size(), mean, std, dataType, randomKey).view(shape);
    }

    static Tensor uniformInt(long startInclusive, long endExclusive, long size, DataType dataType) {
        return uniformInt(startInclusive, endExclusive, size, dataType, nextKey());
    }

    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            long size,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "uniformInt");
        ensureValidRandIntRange(startInclusive, endExclusive, dataType, "uniformInt");

        long range;
        try {
            range = Math.subtractExact(endExclusive, startInclusive);
        } catch (ArithmeticException ex) {
            throw new IllegalArgumentException(
                    "uniformInt range overflow: [" + startInclusive + ", " + endExclusive + ")",
                    ex);
        }

        Tensor uniform = randomUnitInterval(size, DataType.FP64, randomKey, "uniformInt");
        Tensor offset = uniform.multiply((double) range).cast(DataType.I64);
        Tensor shifted = offset.add(startInclusive);
        return dataType == DataType.I64 ? shifted : shifted.cast(dataType);
    }

    static Tensor uniformInt(
            long startInclusive, long endExclusive, Shape shape, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return uniformInt(startInclusive, endExclusive, shape.size(), dataType, nextKey())
                .view(shape);
    }

    static Tensor uniformInt(
            long startInclusive,
            long endExclusive,
            Shape shape,
            DataType dataType,
            RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return uniformInt(startInclusive, endExclusive, shape.size(), dataType, randomKey)
                .view(shape);
    }

    static Tensor normalInt(long size, double mean, double std, DataType dataType) {
        return normalInt(size, mean, std, dataType, nextKey());
    }

    static Tensor normalInt(
            long size, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomInt(dataType, "normalInt");
        ensureValidNormalParams(mean, std, "normalInt");

        Tensor fp = normal(size, mean, std, DataType.FP64, randomKey);
        Tensor rounded =
                fp.greaterThanOrEqual(Tensor.scalar(0.0, DataType.FP64))
                        .where(fp.add(0.5), fp.subtract(0.5))
                        .cast(DataType.I64);
        Tensor clamped =
                rounded.max(Tensor.scalar(randomIntMin(dataType), DataType.I64))
                        .min(Tensor.scalar(randomIntMax(dataType), DataType.I64));
        return dataType == DataType.I64 ? clamped : clamped.cast(dataType);
    }

    static Tensor normalInt(Shape shape, double mean, double std, DataType dataType) {
        Objects.requireNonNull(shape, "shape");
        return normalInt(shape, mean, std, dataType, nextKey());
    }

    static Tensor normalInt(
            Shape shape, double mean, double std, DataType dataType, RandomKey randomKey) {
        Objects.requireNonNull(shape, "shape");
        return normalInt(shape.size(), mean, std, dataType, randomKey).view(shape);
    }

    // endregion Random Creation
    // region Constant / Fill Creation

    static Tensor full(float value, Shape shape) {
        return broadcasted(Float.valueOf(value), DataType.FP32, shape, Device.defaultDevice());
    }

    static Tensor full(double value, Shape shape) {
        return broadcasted(Double.valueOf(value), DataType.FP64, shape, Device.defaultDevice());
    }

    static Tensor full(long value, Shape shape) {
        return broadcasted(Long.valueOf(value), DataType.I64, shape, Device.defaultDevice());
    }

    static Tensor full(int value, Shape shape) {
        return broadcasted(Integer.valueOf(value), DataType.I32, shape, Device.defaultDevice());
    }

    static Tensor full(Number value, DataType dtype, Shape shape) {
        return broadcasted(value, dtype, shape, Device.defaultDevice());
    }

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

    static Tensor scalar(double value, DataType dtype) {
        return broadcasted(Double.valueOf(value), dtype, Shape.scalar(), Device.defaultDevice());
    }

    static Tensor scalar(long value, DataType dtype) {
        return broadcasted(Long.valueOf(value), dtype, Shape.scalar(), Device.defaultDevice());
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
        requireArraySize(data.length, shape);
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyFloatArray(memoryDomain, data, shape);
        return of(view);
    }

    static Tensor of(double[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(double[] data, Shape shape) {
        requireArraySize(data.length, shape);
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyDoubleArray(memoryDomain, data, shape);
        return of(view);
    }

    static Tensor of(int[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(int[] data, Shape shape) {
        requireArraySize(data.length, shape);
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyIntArray(memoryDomain, data, shape);
        return of(view);
    }

    static Tensor of(long[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(long[] data, Shape shape) {
        requireArraySize(data.length, shape);
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyLongArray(memoryDomain, data, shape);
        return of(view);
    }

    static Tensor of(boolean[] data) {
        return of(data, Shape.flat(data.length));
    }

    static Tensor of(boolean[] data, Shape shape) {
        requireArraySize(data.length, shape);
        MemoryDomain<?> memoryDomain =
                Environment.current().runtimeFor(Device.defaultDevice()).memoryDomain();
        MemoryView<?> view = copyBooleanArray(memoryDomain, data, shape);
        return of(view);
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
            long size, DataType dataType, RandomKey randomKey, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);
        Shape shape = shapeFromSize(size, opName);
        Layout layout = Layout.rowMajor(shape);
        RandomComputation computation =
                new RandomComputation(shape, dataType, Device.defaultDevice(), randomKey);
        return lazy(computation, dataType, layout, Device.defaultDevice());
    }

    private static Tensor randomStandardNormal(
            long size, DataType dataType, RandomKey randomKey, String opName) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomKey, "randomKey");
        ensureSupportedRandomFloat(dataType, opName);

        RandomKey k0 = randomKey.split(0L);
        RandomKey k1 = randomKey.split(1L);
        Tensor u1 = randomUnitInterval(size, dataType, k0, opName);
        Tensor u2 = randomUnitInterval(size, dataType, k1, opName);

        if (dataType == DataType.FP64) {
            Tensor r = u2.multiply(-1.0).add(1.0).log().multiply(-2.0).sqrt();
            Tensor theta = u1.multiply(2.0 * Math.PI);
            return theta.cos().multiply(r);
        }

        Tensor r = u2.multiply(-1.0f).add(1.0f).log().multiply(-2.0f).sqrt();
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

    // region Random Key State

    private static RandomKey nextKey() {
        RandomState state = RANDOM_STATE.get();
        long stream = state.counter().getAndIncrement();
        return state.baseKey().split(stream);
    }

    private record RandomState(RandomKey baseKey, AtomicLong counter) {}

    // endregion Random Key State
    // endregion Internal Private Helpers
}

package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.random.RandomAlgorithms;
import com.qxotic.jota.random.RandomKey;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

public final class RandomComputation implements LazyComputation {
    private final Shape shape;
    private final DataType dtype;
    private final Device device;
    private final RandomKey key;

    public RandomComputation(Shape shape, DataType dtype, Device device, RandomKey key) {
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
        this.key = key;
    }

    Shape shape() {
        return shape;
    }

    DataType dataType() {
        return dtype;
    }

    Device device() {
        return device;
    }

    RandomKey key() {
        return key;
    }

    @Override
    public List<Tensor> inputs() {
        return List.of();
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("shape", shape, "dtype", dtype, "device", device.name(), "key", key);
    }

    @Override
    public MemoryView<?> execute() {
        long n = shape.size();
        MemoryDomain<?> memoryDomain = Environment.current().memoryDomainFor(device);
        if (dtype == DataType.FP32) {
            float[] out = new float[Math.toIntExact(n)];
            fillFp32(out);
            return copyFloatArray(memoryDomain, out, shape);
        }
        if (dtype == DataType.FP64) {
            double[] out = new double[Math.toIntExact(n)];
            fillFp64(out);
            return copyDoubleArray(memoryDomain, out, shape);
        }
        throw new IllegalArgumentException("rand supports FP32/FP64 only, got: " + dtype);
    }

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
        return MemoryView.of(dst, DataType.FP32, com.qxotic.jota.Layout.rowMajor(shape));
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
        return MemoryView.of(dst, DataType.FP64, com.qxotic.jota.Layout.rowMajor(shape));
    }

    private void fillFp32(float[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = RandomAlgorithms.uniformFp32(i, key.k0(), key.k1());
        }
    }

    private void fillFp64(double[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = RandomAlgorithms.uniformFp64(i, key.k0(), key.k1());
        }
    }
}

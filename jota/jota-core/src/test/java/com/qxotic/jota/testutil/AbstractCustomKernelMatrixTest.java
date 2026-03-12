package com.qxotic.jota.testutil;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import java.lang.foreign.MemorySegment;

public abstract class AbstractCustomKernelMatrixTest {

    protected record KernelTestSpec(
            String kernelName,
            KernelProgram program,
            LaunchConfig launchConfig,
            Object[] buffers,
            Object[] scalars) {}

    protected abstract void assumeRuntimeReady();

    protected abstract DeviceRuntime createRuntime();

    protected final MemoryDomain<MemorySegment> hostDomain() {
        return DomainFactory.ofMemorySegment();
    }

    protected static <B> MemoryView<B> allocateF32(MemoryDomain<B> domain, Shape shape) {
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(DataType.FP32, shape.size()),
                DataType.FP32,
                Layout.rowMajor(shape));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    protected static void copy(
            MemoryDomain srcDomain, MemoryView src, MemoryDomain dstDomain, MemoryView dst) {
        MemoryDomain.copy(srcDomain, src, dstDomain, dst);
    }

    protected static float readF32(
            MemoryDomain<MemorySegment> host, MemoryView<MemorySegment> view, long index) {
        MemoryAccess<MemorySegment> access = host.directAccess();
        return access.readFloat(view.memory(), view.byteOffset() + index * Float.BYTES);
    }

    protected static void writeF32(
            MemoryDomain<MemorySegment> host,
            MemoryView<MemorySegment> view,
            long index,
            float value) {
        MemoryAccess<MemorySegment> access = host.directAccess();
        access.writeFloat(view.memory(), view.byteOffset() + index * Float.BYTES, value);
    }

    protected static float[] toFlatF32Array(
            MemoryDomain<MemorySegment> host, MemoryView<MemorySegment> view) {
        int size = Math.toIntExact(view.shape().size());
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = readF32(host, view, i);
        }
        return out;
    }

    private static Object[] kernelArgs(Object[] buffers, Object[] scalars) {
        Object[] args = new Object[buffers.length + scalars.length];
        System.arraycopy(buffers, 0, args, 0, buffers.length);
        System.arraycopy(scalars, 0, args, buffers.length, scalars.length);
        return args;
    }

    protected static void registerAndLaunch(DeviceRuntime runtime, KernelTestSpec spec) {
        runtime.registerKernel(spec.kernelName(), spec.program());
        runtime.launchKernel(
                spec.kernelName(), spec.launchConfig(), kernelArgs(spec.buffers(), spec.scalars()));
    }
}

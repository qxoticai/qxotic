package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.random.RandomAlgorithms;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.ExternalToolChecks;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CKernelSmokeTest {

    @Test
    void runsTracedGeluKernel() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv = new Environment(Device.C, current.defaultFloat(), current.runtimes());

        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryView<MemorySegment> inputView =
                createFp32Input(domain, new float[] {-4f, -3f, -2f, -1f, 0f, 1f, 2f, 3f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, Tensor::gelu);
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = domain.directAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < 8; i++) {
            long offset = Indexing.linearToOffset(typed, i);
            float value = access.readFloat(typed.memory(), offset);
            float expected = TestKernels.gelu(i - 4.0f);
            assertEquals(expected, value, 1e-4f);
        }
    }

    @Test
    void runsTracedFp16Kernel() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv = new Environment(Device.C, current.defaultFloat(), current.runtimes());

        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryView<MemorySegment> inputView = createFp16Input(domain, new float[] {1f, 2f, 3f, 4f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, t -> t.add(t));
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = domain.directAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < 4; i++) {
            long offset = Indexing.linearToOffset(typed, i);
            float value = readValue(access, typed, offset);
            float expected = (i + 1.0f) * 2.0f;
            assertEquals(expected, value, 1e-2f);
        }
    }

    @Test
    void runsTracedBf16Kernel() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv = new Environment(Device.C, current.defaultFloat(), current.runtimes());

        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryView<MemorySegment> inputView = createBf16Input(domain, new float[] {1f, 2f, 3f, 4f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, t -> t.add(t));
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = domain.directAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < 4; i++) {
            long offset = Indexing.linearToOffset(typed, i);
            float value = readValue(access, typed, offset);
            float expected = (i + 1.0f) * 2.0f;
            assertEquals(expected, value, 1e-2f);
        }
    }

    @Test
    void runsTracedRandomKernelWithGoldenParity() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv = new Environment(Device.C, current.defaultFloat(), current.runtimes());
        RandomKey key = RandomKey.of(2026L);
        int n = 64;

        MemoryView<?> outFp32 =
                Environment.with(
                        cEnv,
                        () ->
                                Tracer.trace(Tensor.rand(Shape.of(n), DataType.FP32, key), x -> x)
                                        .materialize());
        MemoryView<?> outFp64 =
                Environment.with(
                        cEnv,
                        () ->
                                Tracer.trace(Tensor.rand(Shape.of(n), DataType.FP64, key), x -> x)
                                        .materialize());

        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> fp32 = (MemoryView<MemorySegment>) outFp32;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> fp64 = (MemoryView<MemorySegment>) outFp64;

        for (int i = 0; i < n; i++) {
            long off32 = Indexing.linearToOffset(fp32, i);
            int actual32 = Float.floatToRawIntBits(access.readFloat(fp32.memory(), off32));
            int expected32 =
                    Float.floatToRawIntBits(RandomAlgorithms.uniformFp32(i, key.k0(), key.k1()));
            assertEquals(expected32, actual32);

            long off64 = Indexing.linearToOffset(fp64, i);
            long actual64 = Double.doubleToRawLongBits(access.readDouble(fp64.memory(), off64));
            long expected64 =
                    Double.doubleToRawLongBits(RandomAlgorithms.uniformFp64(i, key.k0(), key.k1()));
            assertEquals(expected64, actual64);
        }
    }

    private static MemoryView<MemorySegment> createFp32Input(
            MemoryDomain<MemorySegment> domain, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        domain.memoryAllocator().allocateMemory(DataType.FP32, values.length),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            access.writeFloat(view.memory(), offset, values[i]);
        }
        return view;
    }

    private static MemoryView<MemorySegment> createFp16Input(
            MemoryDomain<MemorySegment> domain, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        domain.memoryAllocator().allocateMemory(DataType.FP16, values.length),
                        DataType.FP16,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            short bits = Float.floatToFloat16(values[i]);
            access.writeShort(view.memory(), offset, bits);
        }
        return view;
    }

    private static MemoryView<MemorySegment> createBf16Input(
            MemoryDomain<MemorySegment> domain, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        domain.memoryAllocator().allocateMemory(DataType.BF16, values.length),
                        DataType.BF16,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            short bits = BFloat16.fromFloat(values[i]);
            access.writeShort(view.memory(), offset, bits);
        }
        return view;
    }

    private static void assumeCBackendAvailable() {
        Assumptions.assumeTrue(CNative.isAvailable(), "C JNI runtime not available");
        Assumptions.assumeTrue(ExternalToolChecks.hasVersionCommand("gcc"), "gcc not available");
    }

    private static float readValue(
            MemoryAccess<MemorySegment> access, MemoryView<MemorySegment> view, long offset) {
        if (view.dataType() == DataType.FP32) {
            return access.readFloat(view.memory(), offset);
        }
        if (view.dataType() == DataType.FP16) {
            short bits = access.readShort(view.memory(), offset);
            return Float.float16ToFloat(bits);
        }
        if (view.dataType() == DataType.BF16) {
            short bits = access.readShort(view.memory(), offset);
            return BFloat16.toFloat(bits);
        }
        throw new IllegalArgumentException("Unexpected dtype: " + view.dataType());
    }
}

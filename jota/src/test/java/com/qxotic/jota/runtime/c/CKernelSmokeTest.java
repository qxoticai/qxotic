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
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.ExternalToolChecks;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CKernelSmokeTest {

    private static final int FRACTAL_WIDTH = 640;
    private static final int FRACTAL_HEIGHT = 480;
    private static final int FRACTAL_ITERATIONS = 100;

    @Test
    void runsTracedGeluKernel() {
        assumeNotFractalOnlyRun();
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
    void runsTracedMandelbrotKernel() {
        assumeCBackendAvailable();
        assumeFractalsEnabled();

        int width = FRACTAL_WIDTH;
        int height = FRACTAL_HEIGHT;
        int iterations = FRACTAL_ITERATIONS;

        Environment current = Environment.current();
        Environment cEnv = new Environment(Device.C, current.defaultFloat(), current.runtimes());

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced =
                                    Tracer.trace(
                                            List.of(),
                                            inputs ->
                                                    TestKernels.mandelbrotTensor(
                                                            width, height, iterations));
                            return traced.materialize();
                        });

        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;

        TestKernels.writeMandelbrotPpm(
                domain,
                typed,
                Path.of("target", "mandelbrot-c-lir.ppm"),
                width,
                height,
                iterations);

        assertEquals(
                TestKernels.mandelbrotIter(0, 0, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, 0)),
                1e-3f);
        long centerIdx = (long) (height / 2) * width + (width / 2);
        assertEquals(
                TestKernels.mandelbrotIter(height / 2, width / 2, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, centerIdx)),
                1e-3f);
        long lastIdx = (long) (height - 1) * width + (width - 1);
        assertEquals(
                TestKernels.mandelbrotIter(height - 1, width - 1, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, lastIdx)),
                1e-3f);
    }

    @Test
    void runsTracedFp16Kernel() {
        assumeNotFractalOnlyRun();
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
        assumeNotFractalOnlyRun();
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

    private static void assumeFractalsEnabled() {
        Assumptions.assumeTrue(
                Boolean.getBoolean("jota.test.c.fractals") && Boolean.getBoolean("jota.test.ppm"),
                "C fractal smoke tests disabled; set -Djota.test.c.fractals=true and -Djota.test.ppm=true to enable");
    }

    private static void assumeNotFractalOnlyRun() {
        Assumptions.assumeFalse(
                Boolean.getBoolean("jota.test.c.fractals") && Boolean.getBoolean("jota.test.ppm"),
                "Skipping non-fractal C smoke tests in fractal-only run");
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

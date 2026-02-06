package ai.qxotic.jota.c;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import ai.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CKernelSmokeTest {

    private static final int FRACTAL_WIDTH = 64;
    private static final int FRACTAL_HEIGHT = 64;
    private static final int FRACTAL_ITERATIONS = 8;

    @Test
    void runsTracedGeluKernel() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv =
                new Environment(
                        Device.C,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>) Environment.current().backend(Device.C).memoryContext();
        MemoryView<MemorySegment> inputView =
                createFp32Input(context, new float[] {-4f, -3f, -2f, -1f, 0f, 1f, 2f, 3f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, Tensor::gelu);
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = context.memoryAccess();
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
        Environment cEnv =
                new Environment(
                        Device.C,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

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

        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>) Environment.current().backend(Device.C).memoryContext();
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;

        TestKernels.writeMandelbrotPpm(
                context,
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
    void runsTracedPhoenixKernel() {
        assumeCBackendAvailable();
        assumeFractalsEnabled();

        int width = FRACTAL_WIDTH;
        int height = FRACTAL_HEIGHT;
        int iterations = FRACTAL_ITERATIONS;

        Environment current = Environment.current();
        Environment cEnv =
                new Environment(
                        Device.C,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced =
                                    Tracer.trace(
                                            List.of(),
                                            inputs ->
                                                    TestKernels.phoenixTensor(
                                                            width, height, iterations));
                            return traced.materialize();
                        });

        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>) Environment.current().backend(Device.C).memoryContext();
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;

        TestKernels.writePhoenixPpm(
                context,
                typed,
                Path.of("target", "phoenix-c-lir.ppm"),
                width,
                height,
                iterations);

        assertEquals(
                TestKernels.phoenixIter(0, 0, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, 0)),
                1e-3f);
        long centerIdx = (long) (height / 2) * width + (width / 2);
        assertEquals(
                TestKernels.phoenixIter(height / 2, width / 2, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, centerIdx)),
                1e-3f);
        long lastIdx = (long) (height - 1) * width + (width - 1);
        assertEquals(
                TestKernels.phoenixIter(height - 1, width - 1, width, height, iterations),
                access.readFloat(typed.memory(), Indexing.linearToOffset(typed, lastIdx)),
                1e-3f);
    }

    @Test
    void runsTracedFp16Kernel() {
        assumeCBackendAvailable();

        Environment current = Environment.current();
        Environment cEnv =
                new Environment(
                        Device.C,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>) Environment.current().backend(Device.C).memoryContext();
        MemoryView<MemorySegment> inputView =
                createFp16Input(context, new float[] {1f, 2f, 3f, 4f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, t -> t.add(t));
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = context.memoryAccess();
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
        Environment cEnv =
                new Environment(
                        Device.C,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>) Environment.current().backend(Device.C).memoryContext();
        MemoryView<MemorySegment> inputView =
                createBf16Input(context, new float[] {1f, 2f, 3f, 4f});
        Tensor input = Tensor.of(inputView);

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor traced = Tracer.trace(input, t -> t.add(t));
                            return traced.materialize();
                        });

        MemoryAccess<MemorySegment> access = context.memoryAccess();
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < 4; i++) {
            long offset = Indexing.linearToOffset(typed, i);
            float value = readValue(access, typed, offset);
            float expected = (i + 1.0f) * 2.0f;
            assertEquals(expected, value, 1e-2f);
        }
    }

    private static MemoryView<MemorySegment> createFp32Input(
            MemoryContext<MemorySegment> context, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(DataType.FP32, values.length),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            access.writeFloat(view.memory(), offset, values[i]);
        }
        return view;
    }

    private static MemoryView<MemorySegment> createFp16Input(
            MemoryContext<MemorySegment> context, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(DataType.FP16, values.length),
                        DataType.FP16,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            short bits = Float.floatToFloat16(values[i]);
            access.writeShort(view.memory(), offset, bits);
        }
        return view;
    }

    private static MemoryView<MemorySegment> createBf16Input(
            MemoryContext<MemorySegment> context, float[] values) {
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        context.memoryAllocator().allocateMemory(DataType.BF16, values.length),
                        DataType.BF16,
                        Layout.rowMajor(Shape.flat(values.length)));
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(view, i);
            short bits = BFloat16.fromFloat(values[i]);
            access.writeShort(view.memory(), offset, bits);
        }
        return view;
    }

    private static void assumeCBackendAvailable() {
        Assumptions.assumeTrue(CNative.isAvailable(), "C JNI runtime not available");
        try {
            Process process = new ProcessBuilder("gcc", "--version").start();
            int code = process.waitFor();
            Assumptions.assumeTrue(code == 0, "gcc not available");
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "gcc not available");
        }
    }

    private static void assumeFractalsEnabled() {
        Assumptions.assumeTrue(
                Boolean.getBoolean("jota.test.c.fractals")
                        && Boolean.getBoolean("jota.test.ppm"),
                "C fractal smoke tests disabled; set -Djota.test.c.fractals=true and -Djota.test.ppm=true to enable");
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

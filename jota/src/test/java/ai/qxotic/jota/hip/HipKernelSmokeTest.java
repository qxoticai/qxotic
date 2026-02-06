package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Device;
import java.nio.file.Path;
import java.util.List;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import ai.qxotic.jota.testutil.TestKernels;

class HipKernelSmokeTest {

    private static final int FRACTAL_WIDTH = 64;
    private static final int FRACTAL_HEIGHT = 64;
    private static final int FRACTAL_ITERATIONS = 8;

    @Test
    void launchesVecAddKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());

        assumeHipccAvailable();

        int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMemA = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        var hostMemB = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostA =
                MemoryView.of(hostMemA, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostB =
                MemoryView.of(hostMemB, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> hostAccess = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostA, i);
            hostAccess.writeFloat(hostA.memory(), offset, a[i]);
            hostAccess.writeFloat(hostB.memory(), offset, b[i]);
        }

        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devOut =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));

        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(hostA.memory(), hostA.byteOffset(), devA.memory(), 0, byteSize);
        device.memoryOperations()
                .copyFromNative(hostB.memory(), hostB.byteOffset(), devB.memory(), 0, byteSize);

        Tensor aTensor = Tensor.of(devA);
        Tensor bTensor = Tensor.of(devB);
        MemoryView<?> output = aTensor.add(bTensor).materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) output;

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            float value = access.readFloat(hostView.memory(), offset);
            assertEquals(a[i] + b[i], value, 0.0001f);
        }
    }

    @Test
    void launchesLirKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());

        assumeHipccAvailable();

        int n = 32;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i * 0.25f - 2.0f;
        }

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

        Tensor inputTensor = Tensor.of(devInput);
        Tensor traced = Tracer.trace(inputTensor, t -> t.multiply(2.0f).add(1.0f));
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOutput, i);
            float value = access.readFloat(hostOutput.memory(), offset);
            float expected = values[i] * 2.0f + 1.0f;
            assertEquals(expected, value, 1e-4f);
        }
    }

    @Test
    void launchesLirMandelbrotKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();
        assumeFractalsEnabled();

        int width = FRACTAL_WIDTH;
        int height = FRACTAL_HEIGHT;
        int iterations = FRACTAL_ITERATIONS;

        Environment current = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryView<?> output =
                Environment.with(
                        hipEnv,
                        () -> {
                            Tensor traced =
                                    Tracer.trace(
                                            List.of(),
                                            inputs ->
                                                    TestKernels.mandelbrotTensor(
                                                            width, height, iterations));
                            return traced.materialize();
                        });

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);
        MemoryAccess<MemorySegment> access = host.memoryAccess();

        TestKernels.writeMandelbrotPpm(
                host,
                hostOutput,
                Path.of("target", "mandelbrot-hip-lir.ppm"),
                width,
                height,
                iterations);

        assertEquals(
                TestKernels.mandelbrotIter(0, 0, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, 0)),
                1e-3f);
        long centerIdx = (long) (height / 2) * width + (width / 2);
        assertEquals(
                TestKernels.mandelbrotIter(height / 2, width / 2, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, centerIdx)),
                1e-3f);
        long lastIdx = (long) (height - 1) * width + (width - 1);
        assertEquals(
                TestKernels.mandelbrotIter(height - 1, width - 1, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, lastIdx)),
                1e-3f);
    }

    @Test
    void launchesLirPhoenixKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();
        assumeFractalsEnabled();

        int width = FRACTAL_WIDTH;
        int height = FRACTAL_HEIGHT;
        int iterations = FRACTAL_ITERATIONS;

        Environment current = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP,
                        current.defaultFloat(),
                        current.backends(),
                        current.executionMode());

        MemoryView<?> output =
                Environment.with(
                        hipEnv,
                        () -> {
                            Tensor traced =
                                    Tracer.trace(
                                            List.of(),
                                            inputs ->
                                                    TestKernels.phoenixTensor(
                                                            width, height, iterations));
                            return traced.materialize();
                        });

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);
        MemoryAccess<MemorySegment> access = host.memoryAccess();

        TestKernels.writePhoenixPpm(
                host,
                hostOutput,
                Path.of("target", "phoenix-hip-lir.ppm"),
                width,
                height,
                iterations);

        assertEquals(
                TestKernels.phoenixIter(0, 0, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, 0)),
                1e-3f);
        long centerIdx = (long) (height / 2) * width + (width / 2);
        assertEquals(
                TestKernels.phoenixIter(height / 2, width / 2, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, centerIdx)),
                1e-3f);
        long lastIdx = (long) (height - 1) * width + (width - 1);
        assertEquals(
                TestKernels.phoenixIter(height - 1, width - 1, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, lastIdx)),
                1e-3f);
    }

    @Test
    void launchesLirGeluKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i - 4.0f;
        }

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

        Tensor traced = Tracer.trace(Tensor.of(devInput), Tensor::gelu);
        MemoryView<?> output = traced.materialize();
        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);

        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOutput, i);
            float value = hostAccess.readFloat(hostOutput.memory(), offset);
            assertEquals(TestKernels.gelu(values[i]), value, 1e-4f);
        }
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryContext<MemorySegment> host,
            HipMemoryContext device,
            MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) view;
            return hostView;
        }
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> devView = (MemoryView<HipDevicePtr>) view;
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        host.memoryAllocator().allocateMemory(devView.dataType(), devView.shape()),
                        devView.dataType(),
                        devView.layout());
        long byteSize = devView.dataType().byteSizeFor(devView.shape());
        device.memoryOperations()
                .copyToNative(
                        devView.memory(),
                        devView.byteOffset(),
                        hostView.memory(),
                        hostView.byteOffset(),
                        byteSize);
        return hostView;
    }

    private static void assumeHipccAvailable() {
        try {
            Process process = new ProcessBuilder("hipcc", "--version").start();
            int code = process.waitFor();
            Assumptions.assumeTrue(code == 0);
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "hipcc not available");
        }
    }

    private static void assumeFractalsEnabled() {
        Assumptions.assumeTrue(
                Boolean.getBoolean("jota.test.hip.fractals")
                        && Boolean.getBoolean("jota.test.ppm"),
                "HIP fractal smoke tests disabled; set -Djota.test.hip.fractals=true and -Djota.test.ppm=true to enable");
    }
}

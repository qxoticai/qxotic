package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipKernelSmokeTest {

    private static final int FRACTAL_WIDTH = 320;
    private static final int FRACTAL_HEIGHT = 240;
    private static final int FRACTAL_ITERATIONS = 32;

    @Test
    void launchesVecAddKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMemA = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        var hostMemB = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostA =
                MemoryView.of(hostMemA, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostB =
                MemoryView.of(hostMemB, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostA, i);
            hostAccess.writeFloat(hostA.memory(), offset, a[i]);
            hostAccess.writeFloat(hostB.memory(), offset, b[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
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

        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            float value = access.readFloat(hostView.memory(), offset);
            assertEquals(a[i] + b[i], value, 0.0001f);
        }
    }

    @Test
    void launchesLirKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 32;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i * 0.25f - 2.0f;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(
                        hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

        Tensor inputTensor = Tensor.of(devInput);
        Tensor traced = Tracer.trace(inputTensor, t -> t.multiply(2.0f).add(1.0f));
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);
        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOutput, i);
            float value = access.readFloat(hostOutput.memory(), offset);
            float expected = values[i] * 2.0f + 1.0f;
            assertEquals(expected, value, 1e-4f);
        }
    }

    @Test
    void launchesLirMandelbrotKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();
        assumeFractalsEnabled();

        int width = FRACTAL_WIDTH;
        int height = FRACTAL_HEIGHT;
        int iterations = FRACTAL_ITERATIONS;

        Environment current = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP,
                        current.defaultFloat(),
                        current.runtimes(),
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

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<MemorySegment> hostOutput = toHost(host, device, output);
        MemoryAccess<MemorySegment> access = host.directAccess();

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
                access.readFloat(
                        hostOutput.memory(), Indexing.linearToOffset(hostOutput, centerIdx)),
                1e-3f);
        long lastIdx = (long) (height - 1) * width + (width - 1);
        assertEquals(
                TestKernels.mandelbrotIter(height - 1, width - 1, width, height, iterations),
                access.readFloat(hostOutput.memory(), Indexing.linearToOffset(hostOutput, lastIdx)),
                1e-3f);
    }

    @Test
    void launchesLirGeluKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 8;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i - 4.0f;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(
                        hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

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
            MemoryDomain<MemorySegment> host, HipMemoryDomain device, MemoryView<?> view) {
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

    private static void assumeFractalsEnabled() {
        Assumptions.assumeTrue(
                Boolean.getBoolean("jota.test.hip.fractals") && Boolean.getBoolean("jota.test.ppm"),
                "HIP fractal smoke tests disabled; set -Djota.test.hip.fractals=true and -Djota.test.ppm=true to enable");
    }

    // HIP runtime/device assumptions are centralized in HipTestAssumptions.
}

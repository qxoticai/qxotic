package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class MandelbrotTest {

    private static final int WIDTH = 320;
    private static final int HEIGHT = 240;
    private static final int ITERATIONS = 64;

    @Test
    void generatesMandelbrotPpmOnCurrentBackend() {
        Device backend = Environment.current().defaultDevice();
        MemoryView<?> output =
                Tracer.trace(() -> TestKernels.mandelbrotTensor(WIDTH, HEIGHT, ITERATIONS))
                        .materialize();

        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView = toHost(output);
        MemoryAccess<MemorySegment> access = hostDomain.directAccess();

        TestKernels.writeMandelbrotPpm(
                hostDomain,
                hostView,
                Path.of("target", "mandelbrot-" + backend.runtimeId() + "-lir.ppm"),
                WIDTH,
                HEIGHT,
                ITERATIONS);

        assertEquals(
                TestKernels.mandelbrotIter(0, 0, WIDTH, HEIGHT, ITERATIONS),
                access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, 0)),
                1e-3f);
        long centerIdx = (long) (HEIGHT / 2) * WIDTH + (WIDTH / 2);
        assertEquals(
                TestKernels.mandelbrotIter(HEIGHT / 2, WIDTH / 2, WIDTH, HEIGHT, ITERATIONS),
                access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, centerIdx)),
                1e-3f);
        long lastIdx = (long) (HEIGHT - 1) * WIDTH + (WIDTH - 1);
        assertEquals(
                TestKernels.mandelbrotIter(HEIGHT - 1, WIDTH - 1, WIDTH, HEIGHT, ITERATIONS),
                access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, lastIdx)),
                1e-3f);
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        return (MemoryView<MemorySegment>)
                Tensor.of(view).to(Environment.current().nativeRuntime().device()).materialize();
    }
}

package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import com.qxotic.jota.testutil.TestKernels;
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

        TestKernels.writeMandelbrotPpm(
                output,
                Path.of("target", "mandelbrot-" + backend.runtimeId() + "-lir.ppm"),
                WIDTH,
                HEIGHT,
                ITERATIONS);

        Tensor outputTensor = Tensor.of(output);
        assertEquals(
                TestKernels.mandelbrotIter(0, 0, WIDTH, HEIGHT, ITERATIONS),
                TensorTestReads.readFloat(outputTensor, 0),
                1e-3f);
        long centerIdx = (long) (HEIGHT / 2) * WIDTH + (WIDTH / 2);
        assertEquals(
                TestKernels.mandelbrotIter(HEIGHT / 2, WIDTH / 2, WIDTH, HEIGHT, ITERATIONS),
                TensorTestReads.readFloat(outputTensor, centerIdx),
                1e-3f);
        long lastIdx = (long) (HEIGHT - 1) * WIDTH + (WIDTH - 1);
        assertEquals(
                TestKernels.mandelbrotIter(HEIGHT - 1, WIDTH - 1, WIDTH, HEIGHT, ITERATIONS),
                TensorTestReads.readFloat(outputTensor, lastIdx),
                1e-3f);
    }
}

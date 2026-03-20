package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@RunOnAllAvailableBackends
@EnabledIfSystemProperty(named = "jota.test.fractals", matches = "true")
class JuliaSetTest {

    private static final boolean HIGH_QUALITY = Boolean.getBoolean("jota.fractal.highQuality");
    private static final int WIDTH = 320;
    private static final int HEIGHT = 240;
    private static final int ITERATIONS = HIGH_QUALITY ? 192 : 96;
    private static final int TRACE_CHUNK = HIGH_QUALITY ? 32 : 24;

    private static final float X_MIN = -1.8f;
    private static final float X_MAX = 1.8f;
    private static final float Y_MIN = -1.2f;
    private static final float Y_MAX = 1.2f;

    private static final float C_REAL = -0.8f;
    private static final float C_IMAG = 0.156f;

    @Test
    void generatesJuliaPpmOnCurrentBackend() {
        Device backend = Environment.current().defaultDevice();
        MemoryView<?> output =
                juliaTensorChunked(WIDTH, HEIGHT, ITERATIONS, TRACE_CHUNK).materialize();

        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView = toHost(output);
        MemoryAccess<MemorySegment> access = hostDomain.directAccess();

        TestKernels.writePhoenixPpm(
                hostDomain,
                hostView,
                Path.of("target", "julia-" + backend.runtimeId() + "-lir.ppm"),
                WIDTH,
                HEIGHT,
                ITERATIONS);

        assertEquals(
                juliaIter(0, 0, WIDTH, HEIGHT, ITERATIONS),
                access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, 0)),
                1e-3f);
        long center = (long) (HEIGHT / 2) * WIDTH + (WIDTH / 2);
        assertEquals(
                juliaIter(HEIGHT / 2, WIDTH / 2, WIDTH, HEIGHT, ITERATIONS),
                access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, center)),
                1e-3f);
    }

    private static Tensor juliaTensorChunked(int width, int height, int iterations, int chunkSize) {
        Shape shape = Shape.of(height, width);
        float xStep = (X_MAX - X_MIN) / (width - 1);
        float yStep = (Y_MAX - Y_MIN) / (height - 1);

        Tensor zReal =
                Tensor.iota(width, DataType.FP32)
                        .multiply(xStep)
                        .add(X_MIN)
                        .view(Shape.of(1, width))
                        .broadcast(shape);
        Tensor zImag =
                Tensor.iota(height, DataType.FP32)
                        .multiply(yStep)
                        .add(Y_MIN)
                        .view(Shape.of(height, 1))
                        .broadcast(shape);

        Tensor cReal = Tensor.scalar(C_REAL);
        Tensor cImag = Tensor.scalar(C_IMAG);
        Tensor iters = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.BOOL, shape);
        Tensor four = Tensor.scalar(4.0f);

        int start = 0;
        while (start < iterations) {
            int endExclusive = Math.min(start + chunkSize, iterations);
            for (int i = start; i < endExclusive; i++) {
                Tensor zRealNew = zReal.square().subtract(zImag.square()).add(cReal);
                Tensor zImagNew = zReal.multiply(zImag).multiply(2.0f).add(cImag);

                Tensor magnitude2 = zRealNew.square().add(zImagNew.square());
                Tensor hasEscaped = magnitude2.greaterThan(four);
                Tensor justEscaped = hasEscaped.logicalAnd(escaped.logicalNot());

                Tensor iterValue = Tensor.full((float) i, DataType.FP32, shape);
                iters = justEscaped.where(iterValue, iters);
                escaped = escaped.logicalOr(hasEscaped);
                zReal = zRealNew;
                zImag = zImagNew;
            }
            zReal = Tensor.of(zReal.materialize());
            zImag = Tensor.of(zImag.materialize());
            iters = Tensor.of(iters.materialize());
            escaped = Tensor.of(escaped.materialize());
            start = endExclusive;
        }

        Tensor finalIter = Tensor.full((float) (iterations - 1), DataType.FP32, shape);
        return escaped.where(iters, finalIter);
    }

    private static float juliaIter(int row, int col, int width, int height, int iterations) {
        float xStep = (X_MAX - X_MIN) / (width - 1);
        float yStep = (Y_MAX - Y_MIN) / (height - 1);
        float zReal = X_MIN + col * xStep;
        float zImag = Y_MIN + row * yStep;
        for (int i = 0; i < iterations; i++) {
            float nextReal = zReal * zReal - zImag * zImag + C_REAL;
            float nextImag = 2.0f * zReal * zImag + C_IMAG;
            float mag2 = nextReal * nextReal + nextImag * nextImag;
            if (mag2 > 4.0f) {
                return i;
            }
            zReal = nextReal;
            zImag = nextImag;
        }
        return iterations - 1;
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        return (MemoryView<MemorySegment>)
                Tensor.of(view).to(Environment.current().nativeRuntime().device()).materialize();
    }
}

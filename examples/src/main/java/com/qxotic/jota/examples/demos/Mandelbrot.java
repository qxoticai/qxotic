package com.qxotic.jota.examples.demos;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;

/** Mandelbrot set visualization using Jota's tensor API. */
public final class Mandelbrot {

    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final int MAX_ITER = 100;

    private static final float X_MIN = -2.05f;
    private static final float X_MAX = 0.60f;
    private static final float Y_MIN = -1.15f;
    private static final float Y_MAX = 1.15f;

    private Mandelbrot() {}

    public static void main(String[] args) throws IOException {
        Environment current = Environment.current();
        if (DemoDevices.hasListDevicesFlag(args)) {
            System.out.println(DemoDevices.listDevices(current));
            return;
        }

        Device backend = DemoDevices.resolveDevice(current, requestedBackend(args));
        Environment backendEnv =
                Environment.of(backend, current.defaultFloat(), current.runtimes());

        long start = System.currentTimeMillis();
        byte[] rgb =
                Environment.with(
                        backendEnv,
                        () -> {
                            Tensor iterations =
                                    Tracer.trace(List.of(), inputs -> computeMandelbrot());
                            return toRGB(iterations);
                        });
        long elapsed = System.currentTimeMillis() - start;
        System.out.println(
                "Backend="
                        + backend.runtimeId()
                        + ", runtime="
                        + elapsed
                        + "ms"
                        + " ("
                        + WIDTH
                        + "x"
                        + HEIGHT
                        + ", MAX_ITER="
                        + MAX_ITER
                        + ")");
        String filename = "mandelbrot-" + backend.runtimeId() + ".ppm";
        PpmWriter.write(Path.of(filename), WIDTH, HEIGHT, rgb);
    }

    private static Tensor linspace(float start, float end, int count) {
        float step = (end - start) / (count - 1);
        return Tensor.iota(count, DataType.FP32).multiply(step).add(start);
    }

    private static Tensor computeMandelbrot() {
        Shape image = Shape.of(HEIGHT, WIDTH);
        Tensor cReal = linspace(X_MIN, X_MAX, WIDTH).view(Shape.of(1, WIDTH)).broadcast(image);
        Tensor cImag = linspace(Y_MIN, Y_MAX, HEIGHT).view(Shape.of(HEIGHT, 1)).broadcast(image);

        Tensor zReal = Tensor.zeros(DataType.FP32, image);
        Tensor zImag = Tensor.zeros(DataType.FP32, image);
        Tensor iterations = Tensor.zeros(DataType.FP32, image);
        Tensor escaped = Tensor.zeros(DataType.BOOL, image);

        for (int i = 0; i < MAX_ITER; i++) {
            Tensor zRealNext = zReal.square().subtract(zImag.square()).add(cReal);
            Tensor zImagNext = zReal.multiply(zImag).multiply(2f).add(cImag);
            Tensor escapedNow =
                    zRealNext.square().add(zImagNext.square()).greaterThan(Tensor.scalar(4f));
            Tensor justEscaped = escapedNow.logicalAnd(escaped.logicalNot());
            iterations = justEscaped.where(Tensor.full(i, DataType.FP32, image), iterations);
            escaped = escaped.logicalOr(escapedNow);
            zReal = zRealNext;
            zImag = zImagNext;
        }

        return escaped.where(iterations, Tensor.full(MAX_ITER - 1, DataType.FP32, image));
    }

    @SuppressWarnings("unchecked")
    private static byte[] toRGB(Tensor iterations) {
        MemoryView<MemorySegment> typedView = toNativeHostView(iterations.materialize());
        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>) Environment.nativeRuntime().memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();

        byte[] rgb = new byte[HEIGHT * WIDTH * 3];

        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                int idx = h * WIDTH + w;
                long offset = Indexing.linearToOffset(typedView, idx);
                float iter = access.readFloat(typedView.memory(), offset);

                if (iter >= MAX_ITER - 1) {
                    rgb[idx * 3] = 0;
                    rgb[idx * 3 + 1] = 0;
                    rgb[idx * 3 + 2] = 0;
                } else {
                    double t = iter / MAX_ITER;
                    rgb[idx * 3] = (byte) (9 * (1 - t) * t * t * t * 255);
                    rgb[idx * 3 + 1] = (byte) (15 * (1 - t) * (1 - t) * t * t * 255);
                    rgb[idx * 3 + 2] = (byte) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
                }
            }
        }

        return rgb;
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toNativeHostView(MemoryView<?> view) {
        return (MemoryView<MemorySegment>)
                Tensor.of(view).to(Environment.nativeRuntime().device()).materialize();
    }

    private static String requestedBackend(String[] args) {
        for (String arg : args) {
            if (arg == null || arg.isBlank()) {
                continue;
            }
            if ("--list-devices".equalsIgnoreCase(arg)) {
                continue;
            }
            if (arg.startsWith("--backend=")) {
                return arg.substring("--backend=".length());
            }
            if (!arg.startsWith("--")) {
                return arg;
            }
        }
        return null;
    }
}

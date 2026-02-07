package ai.qxotic.jota.examples;

import ai.qxotic.jota.*;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * Mandelbrot set visualization using Jota's tensor API.
 *
 * <p>This demo computes the Mandelbrot set using a fully tensorized approach - all pixels are
 * computed in parallel using tensor operations rather than per-pixel loops.
 */
public class MandelbrotDemo {

    // Image dimensions
    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;

    // Complex plane bounds (centered on interesting region)
    private static final float X_MIN = -2.5f;
    private static final float X_MAX = 1.0f;
    private static final float Y_MIN = -1.25f;
    private static final float Y_MAX = 1.25f;

    // Maximum iterations before assuming point is in the set
    private static final int MAX_ITER = 100;

    public static void main(String[] args) throws IOException {
        Device backend = resolveBackend(args);
        if (!Environment.current().runtimes().hasRuntime(backend)) {
            throw new IllegalStateException("Backend runtime is not available: " + backend);
        }

        Environment current = Environment.current();
        Environment backendEnv =
                new Environment(
                        backend,
                        current.defaultFloat(),
                        current.runtimes(),
                        current.executionMode());

        long start = System.currentTimeMillis();
        int[][] rgb =
                Environment.with(
                        backendEnv,
                        () -> {
                            Tensor iterations =
                                    Tracer.trace(
                                            List.of(), inputs -> computeMandelbrotPureTensor());
                            return toRGB(iterations);
                        });
        long elapsed = System.currentTimeMillis() - start;

        System.out.println(
                "Backend="
                        + backend.name()
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

        String filename = "mandelbrot-" + backend.leafName() + ".ppm";
        writePPM(filename, rgb, WIDTH, HEIGHT);
    }

    @Test
    void generateMandelbrot() throws IOException {
        main(new String[0]);
    }

    /**
     * Creates a 1D tensor with evenly spaced values (like numpy.linspace).
     *
     * @param start starting value
     * @param end ending value (inclusive)
     * @param count number of samples
     * @return 1D tensor of shape [count] with evenly spaced values
     */
    private static Tensor linspace(float start, float end, int count) {
        // linspace(start, end, count) = start + arange(count) * (end - start) / (count - 1)
        float step = (end - start) / (count - 1);
        return Tensor.iota(count, DataType.FP32).multiply(step).add(start);
    }

    /**
     * Computes Mandelbrot using pure tensor operations.
     *
     * <p><b>NOTE:</b> This approach builds a dataflow graph where each iteration adds new nodes.
     * For N iterations, the graph size is O(N), which causes problems for large N:
     *
     * <ul>
     *   <li>Graph construction becomes slow
     *   <li>Lowering to LIR produces huge output
     *   <li>Java source code generation may fail or produce uncompilable code
     * </ul>
     *
     * <p>The fundamental issue is that TIR doesn't have loop constructs - it represents
     * computations as a DAG. Iterative algorithms like Mandelbrot need either:
     *
     * <ul>
     *   <li>Loop constructs in TIR (not yet implemented)
     *   <li>Manual kernel construction with LIR loops (see MandelbrotKernel)
     *   <li>Materializing intermediate results between traces
     * </ul>
     *
     * <p>For small MAX_ITER (e.g., 8), this works fine. For larger values, use MandelbrotKernel.
     */
    private static Tensor computeMandelbrotPureTensor() {
        Shape shape = Shape.of(HEIGHT, WIDTH);

        float xStep = (X_MAX - X_MIN) / (WIDTH - 1);
        float yStep = (Y_MAX - Y_MIN) / (HEIGHT - 1);
        Tensor xCoords =
                Tensor.iota(WIDTH, DataType.FP32)
                        .multiply(xStep)
                        .add(X_MIN)
                        .view(Shape.of(1, WIDTH));
        Tensor yCoords =
                Tensor.iota(HEIGHT, DataType.FP32)
                        .multiply(yStep)
                        .add(Y_MIN)
                        .view(Shape.of(HEIGHT, 1));
        Tensor cReal = xCoords.broadcast(Shape.of(HEIGHT, WIDTH));
        Tensor cImag = yCoords.broadcast(Shape.of(HEIGHT, WIDTH));

        Tensor zReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zImag = Tensor.zeros(DataType.FP32, shape);
        Tensor iterations = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.BOOL, shape);

        Tensor four = Tensor.scalar(4.0f);

        for (int i = 0; i < MAX_ITER; i++) {
            Tensor zReal2 = zReal.square();
            Tensor zImag2 = zImag.square();
            Tensor zRealNew = zReal2.subtract(zImag2).add(cReal);
            Tensor zImagNew = zReal.multiply(zImag).multiply(2.0f).add(cImag);

            Tensor magnitude2 = zRealNew.square().add(zImagNew.square());
            Tensor hasEscaped = magnitude2.greaterThan(four);
            Tensor notYetEscaped = escaped.logicalNot();
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            Tensor iterValue = Tensor.full((float) i, DataType.FP32, shape);
            iterations = Tensor.where(justEscaped, iterValue, iterations);

            escaped = escaped.logicalOr(hasEscaped);

            zReal = zRealNew;
            zImag = zImagNew;
        }

        Tensor finalIter = Tensor.full((float) (MAX_ITER - 1), DataType.FP32, shape);
        iterations = Tensor.where(escaped, iterations, finalIter);

        return iterations;
    }

    private static float[] computeMandelbrotJava() {
        float[] iterations = new float[HEIGHT * WIDTH];
        boolean[] escaped = new boolean[HEIGHT * WIDTH];

        float[] xCoords = linspaceJava(X_MIN, X_MAX, WIDTH);
        float[] yCoords = linspaceJava(Y_MIN, Y_MAX, HEIGHT);

        float[] zReal = new float[HEIGHT * WIDTH];
        float[] zImag = new float[HEIGHT * WIDTH];

        for (int i = 0; i < MAX_ITER; i++) {
            for (int h = 0; h < HEIGHT; h++) {
                float cImag = yCoords[h];
                int rowBase = h * WIDTH;
                for (int w = 0; w < WIDTH; w++) {
                    int idx = rowBase + w;
                    float cReal = xCoords[w];

                    float zr = zReal[idx];
                    float zi = zImag[idx];
                    float zr2 = zr * zr;
                    float zi2 = zi * zi;

                    float zrNew = zr2 - zi2 + cReal;
                    float ziNew = 2.0f * zr * zi + cImag;

                    zReal[idx] = zrNew;
                    zImag[idx] = ziNew;

                    float magnitude2 = zrNew * zrNew + ziNew * ziNew;
                    if (!escaped[idx] && magnitude2 > 4.0f) {
                        iterations[idx] = i;
                        escaped[idx] = true;
                    }
                }
            }
        }

        for (int i = 0; i < iterations.length; i++) {
            if (!escaped[i]) {
                iterations[i] = MAX_ITER - 1;
            }
        }

        return iterations;
    }

    private static float[] linspaceJava(float start, float end, int count) {
        float[] values = new float[count];
        if (count == 1) {
            values[0] = start;
            return values;
        }
        float step = (end - start) / (count - 1);
        for (int i = 0; i < count; i++) {
            values[i] = start + step * i;
        }
        return values;
    }

    /**
     * Converts iteration counts to RGB colors.
     *
     * @param iterations tensor of iteration counts
     * @return 2D array [HEIGHT*WIDTH][3] of RGB values (0-255)
     */
    @SuppressWarnings("unchecked")
    private static int[][] toRGB(Tensor iterations) {
        MemoryView<MemorySegment> typedView = toNativeHostView(iterations.materialize());
        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();

        // Check iteration value range
        float minIter = Float.MAX_VALUE;
        float maxIter = Float.MIN_VALUE;
        for (int i = 0; i < HEIGHT * WIDTH; i++) {
            long off = Indexing.linearToOffset(typedView, i);
            float val = access.readFloat(typedView.memory(), off);
            minIter = Math.min(minIter, val);
            maxIter = Math.max(maxIter, val);
        }
        System.out.println("Iteration range: " + minIter + " to " + maxIter);

        int[][] rgb = new int[HEIGHT * WIDTH][3];

        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                int idx = h * WIDTH + w;
                long offset = Indexing.linearToOffset(typedView, idx);
                float iter = access.readFloat(typedView.memory(), offset);

                if (iter >= MAX_ITER - 1) {
                    // Point is in the Mandelbrot set - color it black
                    rgb[idx][0] = 0;
                    rgb[idx][1] = 0;
                    rgb[idx][2] = 0;
                } else {
                    // Color based on iteration count (smooth coloring)
                    double t = iter / MAX_ITER;
                    rgb[idx][0] = (int) (9 * (1 - t) * t * t * t * 255);
                    rgb[idx][1] = (int) (15 * (1 - t) * (1 - t) * t * t * 255);
                    rgb[idx][2] = (int) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
                }
            }
        }

        return rgb;
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toNativeHostView(MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment
                && view.memory().device().belongsTo(Device.CPU)) {
            return (MemoryView<MemorySegment>) view;
        }
        MemoryDomain<MemorySegment> hostDomain =
                (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        hostDomain.memoryAllocator().allocateMemory(view.dataType(), view.shape()),
                        view.dataType(),
                        view.layout());
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryDomain.copy(srcDomain, srcView, hostDomain, hostView);
        return hostView;
    }

    private static Device resolveBackend(String[] args) {
        String requested = null;
        for (String arg : args) {
            if (arg == null || arg.isBlank()) {
                continue;
            }
            if (arg.startsWith("--backend=")) {
                requested = arg.substring("--backend=".length());
                break;
            }
            requested = arg;
            break;
        }
        if (requested == null || requested.isBlank() || requested.equalsIgnoreCase("native")) {
            return Environment.current().nativeRuntime().device();
        }
        String normalized = requested.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "panama", "native-panama" -> Device.PANAMA;
            case "java-aot" -> Device.JAVA_AOT;
            case "c" -> Device.C;
            case "hip" -> Device.HIP;
            default ->
                    throw new IllegalArgumentException(
                            "Unknown backend '"
                                    + requested
                                    + "'. Use one of: native, panama, java-aot, c, hip");
        };
    }

    private static int[][] toRGB(float[] iterations) {
        int[][] rgb = new int[HEIGHT * WIDTH][3];

        float minIter = Float.MAX_VALUE;
        float maxIter = Float.MIN_VALUE;
        for (float iter : iterations) {
            minIter = Math.min(minIter, iter);
            maxIter = Math.max(maxIter, iter);
        }
        System.out.println("Java iteration range: " + minIter + " to " + maxIter);

        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                int idx = h * WIDTH + w;
                float iter = iterations[idx];

                if (iter >= MAX_ITER - 1) {
                    rgb[idx][0] = 0;
                    rgb[idx][1] = 0;
                    rgb[idx][2] = 0;
                } else {
                    double t = iter / MAX_ITER;
                    rgb[idx][0] = (int) (9 * (1 - t) * t * t * t * 255);
                    rgb[idx][1] = (int) (15 * (1 - t) * (1 - t) * t * t * 255);
                    rgb[idx][2] = (int) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
                }
            }
        }

        return rgb;
    }

    /**
     * Writes an image in PPM (Portable Pixmap) format.
     *
     * @param filename output filename
     * @param rgb pixel colors as [index][r,g,b]
     * @param width image width
     * @param height image height
     */
    private static void writePPM(String filename, int[][] rgb, int width, int height)
            throws IOException {
        try (PrintStream out =
                new PrintStream(new BufferedOutputStream(new FileOutputStream(filename)))) {
            // PPM header
            out.println("P3");
            out.println(width + " " + height);
            out.println("255");

            // Pixel data
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = h * width + w;
                    out.print(rgb[idx][0] + " " + rgb[idx][1] + " " + rgb[idx][2]);
                    if (w < width - 1) {
                        out.print(" ");
                    }
                }
                out.println();
            }
        }
    }
}

package ai.qxotic.jota.examples;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Mandelbrot set visualization using Jota's tensor API.
 *
 * <p>This demo computes the Mandelbrot set using a fully tensorized approach - all pixels are
 * computed in parallel using tensor operations rather than per-pixel loops.
 */
public class MandelbrotDemo {

    // Image dimensions
    private static final int WIDTH = 800 / 4;
    private static final int HEIGHT = 600 / 4;

    // Complex plane bounds (centered on interesting region)
    private static final float X_MIN = -2.5f;
    private static final float X_MAX = 1.0f;
    private static final float Y_MIN = -1.25f;
    private static final float Y_MAX = 1.25f;

    // Maximum iterations before assuming point is in the set
    private static final int MAX_ITER = 100;

    public static void main(String[] args) throws IOException {
        long startTime = System.currentTimeMillis();

        // Compute Mandelbrot iteration counts
        Tensor iterations = computeMandelbrot();

        // Convert to RGB colors
        int[][] rgb = toRGB(iterations);

        // Write PPM image
        String filename = "mandelbrot.ppm";
        writePPM(filename, rgb, WIDTH, HEIGHT);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println(
                "Generated " + filename + " (" + WIDTH + "x" + HEIGHT + ") in " + elapsed + "ms");
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
        return Tensor.arange(count, DataType.FP32).multiply(step).add(start);
    }

    /**
     * Computes the Mandelbrot set using tensorized operations.
     *
     * @return Tensor of shape [HEIGHT, WIDTH] containing iteration counts
     */
    private static Tensor computeMandelbrot() {
        Shape shape = Shape.of(HEIGHT, WIDTH);

        // Create coordinate grids
        // real: 1D tensor of x coordinates, then broadcast to 2D
        // imag: 1D tensor of y coordinates, then broadcast to 2D
        Tensor real1D = linspace(X_MIN, X_MAX, WIDTH); // Shape: [WIDTH]
        Tensor imag1D = linspace(Y_MIN, Y_MAX, HEIGHT); // Shape: [HEIGHT]

        // Broadcast to 2D: cReal[h,w] = real1D[w], cImag[h,w] = imag1D[h]
        // Real: expand to [1, WIDTH] then broadcast to [HEIGHT, WIDTH]
        Tensor cReal = real1D.view(Shape.of(1, WIDTH)).broadcast(shape);
        // Imag: expand to [HEIGHT, 1] then broadcast to [HEIGHT, WIDTH]
        Tensor cImag = imag1D.view(Shape.of(HEIGHT, 1)).broadcast(shape);

        // Initialize z = 0 + 0i
        Tensor zReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zImag = Tensor.zeros(DataType.FP32, shape);

        // Track iteration counts and escape status
        Tensor iterations = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.FP32, shape); // 0.0 = not escaped, 1.0 = escaped

        // Iterate: z = z² + c
        for (int i = 0; i < MAX_ITER; i++) {
            // z² = (a + bi)² = (a² - b²) + (2ab)i
            Tensor zReal2 = zReal.square();
            Tensor zImag2 = zImag.square();
            Tensor zRealNew = zReal2.subtract(zImag2).add(cReal);
            Tensor zImagNew = zReal.multiply(zImag).multiply(2.0f).add(cImag);

            // |z|² = zReal² + zImag²
            Tensor magnitude2 = zRealNew.square().add(zImagNew.square());

            // Points escape when |z|² > 4
            // justEscaped = (magnitude² > 4) AND (NOT escaped)
            Tensor hasEscaped = magnitude2.greaterThan(Tensor.scalar(4.0f));
            Tensor notYetEscaped = escaped.lessThan(Tensor.scalar(0.5f));
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            // Record iteration count for newly escaped points
            // iterations = where(justEscaped, i, iterations)
            Tensor iterValue = Tensor.broadcasted((float) i, shape);
            iterations = Tensor.where(justEscaped, iterValue, iterations);

            // Update escaped mask
            // escaped = where(justEscaped, 1.0, escaped)
            Tensor one = Tensor.broadcasted(1.0f, shape);
            escaped = Tensor.where(justEscaped, one, escaped);

            // Update z
            zReal = zRealNew;
            zImag = zImagNew;

            // Checkpoint every 10 iterations to prevent expression graph explosion
            //            if (i % 10 == 9) {
            //                zReal = Tensor.of(zReal.materialize());
            //                zImag = Tensor.of(zImag.materialize());
            //                iterations = Tensor.of(iterations.materialize());
            //                escaped = Tensor.of(escaped.materialize());
            //            }
        }

        return iterations;
    }

    /**
     * Converts iteration counts to RGB colors.
     *
     * @param iterations tensor of iteration counts
     * @return 2D array [HEIGHT*WIDTH][3] of RGB values (0-255)
     */
    @SuppressWarnings("unchecked")
    private static int[][] toRGB(Tensor iterations) {
        MemoryView<?> view = iterations.materialize();
        MemoryContext<MemorySegment> context =
                (MemoryContext<MemorySegment>)
                        Environment.current().nativeBackend().memoryContext();
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = context.memoryAccess();

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

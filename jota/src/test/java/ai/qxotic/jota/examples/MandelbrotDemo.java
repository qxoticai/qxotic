package ai.qxotic.jota.examples;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
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
        //        long tensorStart = System.currentTimeMillis();
        //        Tensor iterations = computeMandelbrot();
        //        int[][] rgb = toRGB(iterations);
        //        long tensorElapsed = System.currentTimeMillis() - tensorStart;

        long pureTensorStart = System.currentTimeMillis();
        Tensor pureIterations = Tracer.trace(List.of(), inputs -> computeMandelbrotPureTensor());
        int[][] pureRgb = toRGB(pureIterations);
        long pureTensorElapsed = System.currentTimeMillis() - pureTensorStart;

        System.out.println(
                "Tensor runtime: "
                        // + tensorElapsed
                        + "ms, Pure Tensor runtime: "
                        + pureTensorElapsed
                        + "ms"
                        + " ("
                        + WIDTH
                        + "x"
                        + HEIGHT
                        + ", MAX_ITER="
                        + MAX_ITER
                        + ")");

        pureTensorStart = System.currentTimeMillis();
        pureIterations = Tracer.trace(List.of(), inputs -> computeMandelbrotPureTensor());
        pureRgb = toRGB(pureIterations);
        pureTensorElapsed = System.currentTimeMillis() - pureTensorStart;

        System.out.println(
                "Tensor runtime: "
                        // + tensorElapsed
                        + "ms, Pure Tensor runtime: "
                        + pureTensorElapsed
                        + "ms"
                        + " ("
                        + WIDTH
                        + "x"
                        + HEIGHT
                        + ", MAX_ITER="
                        + MAX_ITER
                        + ")");

        // Write output image
        String filename = "mandelbrot.ppm";
        writePPM(filename, pureRgb, WIDTH, HEIGHT);
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

    //
    //    private record MandelbrotKernel(LIRGraph graph, ScratchLayout scratchLayout) {
    //        static MandelbrotKernel compile(Shape shape) {
    //            LIRGraph graph = buildGraph(shape);
    //            LIRGraph optimized = new LIRStandardPipeline().run(graph);
    //            ScratchLayout scratchLayout = new ScratchAnalysisPass().analyze(optimized);
    //            return new MandelbrotKernel(optimized, scratchLayout);
    //        }
    //
    //        private static LIRGraph buildGraph(Shape shape) {
    //            LIRGraph.Builder builder = LIRGraph.builder();
    //            BufferRef zReal = builder.addContiguousInput(DataType.FP32, shape.toArray());
    //            BufferRef zImag = builder.addContiguousInput(DataType.FP32, shape.toArray());
    //            BufferRef cReal = builder.addContiguousInput(DataType.FP32, shape.toArray());
    //            BufferRef cImag = builder.addContiguousInput(DataType.FP32, shape.toArray());
    //            BufferRef escaped = builder.addContiguousInput(DataType.BOOL, shape.toArray());
    //            BufferRef iterations = builder.addContiguousInput(DataType.FP32, shape.toArray());
    //
    //            BufferRef zRealOut = builder.addContiguousOutput(DataType.FP32, shape.toArray());
    //            BufferRef zImagOut = builder.addContiguousOutput(DataType.FP32, shape.toArray());
    //            BufferRef escapedOut = builder.addContiguousOutput(DataType.BOOL,
    // shape.toArray());
    //            BufferRef iterationsOut = builder.addContiguousOutput(DataType.FP32,
    // shape.toArray());
    //
    //            IndexVar i0 = new IndexVar("i0");
    //            IndexVar i1 = new IndexVar("i1");
    //            IndexVar k = new IndexVar("k");
    //
    //            IndexExpr offZReal = offset2d(zReal, i0, i1);
    //            IndexExpr offZImag = offset2d(zImag, i0, i1);
    //            IndexExpr offCReal = offset2d(cReal, i0, i1);
    //            IndexExpr offCImag = offset2d(cImag, i0, i1);
    //            IndexExpr offEscaped = offset2d(escaped, i0, i1);
    //            IndexExpr offIterations = offset2d(iterations, i0, i1);
    //            IndexExpr offZRealOut = offset2d(zRealOut, i0, i1);
    //            IndexExpr offZImagOut = offset2d(zImagOut, i0, i1);
    //            IndexExpr offEscapedOut = offset2d(escapedOut, i0, i1);
    //            IndexExpr offIterationsOut = offset2d(iterationsOut, i0, i1);
    //
    //            ScalarExpr zrInit = new ScalarLoad(zReal, offZReal);
    //            ScalarExpr ziInit = new ScalarLoad(zImag, offZImag);
    //            ScalarExpr cr = new ScalarLoad(cReal, offCReal);
    //            ScalarExpr ci = new ScalarLoad(cImag, offCImag);
    //            ScalarExpr escInit = new ScalarLoad(escaped, offEscaped);
    //            ScalarExpr iterInit = new ScalarLoad(iterations, offIterations);
    //
    //            ScalarRef zr = new ScalarRef("zr", DataType.FP32);
    //            ScalarRef zi = new ScalarRef("zi", DataType.FP32);
    //            ScalarRef esc = new ScalarRef("esc", DataType.BOOL);
    //            ScalarRef iter = new ScalarRef("iter", DataType.FP32);
    //
    //            ScalarExpr zr2 = new ScalarBinary(BinaryOperator.MULTIPLY, zr, zr);
    //            ScalarExpr zi2 = new ScalarBinary(BinaryOperator.MULTIPLY, zi, zi);
    //            ScalarExpr zrNewExpr =
    //                    new ScalarBinary(
    //                            BinaryOperator.ADD,
    //                            new ScalarBinary(BinaryOperator.SUBTRACT, zr2, zi2),
    //                            cr);
    //            ScalarExpr ziMul = new ScalarBinary(BinaryOperator.MULTIPLY, zr, zi);
    //            ScalarExpr ziNewExpr =
    //                    new ScalarBinary(
    //                            BinaryOperator.ADD,
    //                            new ScalarBinary(
    //                                    BinaryOperator.MULTIPLY,
    //                                    ziMul,
    //                                    ScalarLiteral.ofFloat(2.0f)),
    //                            ci);
    //
    //            ScalarLet zrNewLet = new ScalarLet("zrNew", zrNewExpr);
    //            ScalarLet ziNewLet = new ScalarLet("ziNew", ziNewExpr);
    //            ScalarRef zrNew = new ScalarRef("zrNew", DataType.FP32);
    //            ScalarRef ziNew = new ScalarRef("ziNew", DataType.FP32);
    //
    //            ScalarExpr mag2Expr =
    //                    new ScalarBinary(
    //                            BinaryOperator.ADD,
    //                            new ScalarBinary(BinaryOperator.MULTIPLY, zrNew, zrNew),
    //                            new ScalarBinary(BinaryOperator.MULTIPLY, ziNew, ziNew));
    //            ScalarLet mag2Let = new ScalarLet("mag2", mag2Expr);
    //            ScalarRef mag2 = new ScalarRef("mag2", DataType.FP32);
    //
    //            ScalarExpr hasEscapedExpr =
    //                    new ScalarBinary(
    //                            BinaryOperator.LESS_THAN, ScalarLiteral.ofFloat(4.0f), mag2);
    //            ScalarLet hasEscapedLet = new ScalarLet("hasEscaped", hasEscapedExpr);
    //            ScalarRef hasEscaped = new ScalarRef("hasEscaped", DataType.BOOL);
    //
    //            ScalarExpr notYetExpr = new ScalarUnary(UnaryOperator.LOGICAL_NOT, esc);
    //            ScalarLet notYetLet = new ScalarLet("notYet", notYetExpr);
    //            ScalarRef notYet = new ScalarRef("notYet", DataType.BOOL);
    //
    //            ScalarExpr justEscapedExpr =
    //                    new ScalarBinary(BinaryOperator.LOGICAL_AND, hasEscaped, notYet);
    //            ScalarLet justEscapedLet = new ScalarLet("justEscaped", justEscapedExpr);
    //            ScalarRef justEscaped = new ScalarRef("justEscaped", DataType.BOOL);
    //
    //            ScalarExpr iterIndex = new ScalarFromIndex(k);
    //            ScalarExpr iterValue = new ScalarCast(iterIndex, DataType.FP32);
    //            ScalarExpr iterOut = new ScalarTernary(justEscaped, iterValue, iter);
    //            ScalarExpr escOut = new ScalarTernary(justEscaped, ScalarLiteral.ofBool(true),
    // esc);
    //
    //            Yield yield =
    //                    new Yield(List.of(zrNew, ziNew, escOut, iterOut));
    //            Block innerBody =
    //                    Block.of(
    //                            zrNewLet,
    //                            ziNewLet,
    //                            mag2Let,
    //                            hasEscapedLet,
    //                            notYetLet,
    //                            justEscapedLet,
    //                            yield);
    //            StructuredFor iterLoop =
    //                    StructuredFor.of(
    //                            "k",
    //                            0,
    //                            MAX_ITER,
    //                            1,
    //                            List.of(
    //                                    new LoopIterArg("zr", DataType.FP32, zrInit),
    //                                    new LoopIterArg("zi", DataType.FP32, ziInit),
    //                                    new LoopIterArg("esc", DataType.BOOL, escInit),
    //                                    new LoopIterArg("iter", DataType.FP32, iterInit)),
    //                            innerBody);
    //
    //            Store storeZReal = new Store(zRealOut, offZRealOut, zr);
    //            Store storeZImag = new Store(zImagOut, offZImagOut, zi);
    //            Store storeIterations = new Store(iterationsOut, offIterationsOut, iter);
    //            Store storeEscaped = new Store(escapedOut, offEscapedOut, esc);
    //
    //            Block body = Block.of(iterLoop, storeZReal, storeZImag, storeIterations,
    // storeEscaped);
    //            StructuredFor inner = StructuredFor.of("i1", 0, WIDTH, 1, List.of(), body);
    //            StructuredFor outer = StructuredFor.of("i0", 0, HEIGHT, 1, List.of(), inner);
    //            return builder.build(outer);
    //        }
    //
    //        private static IndexExpr offset2d(BufferRef buffer, IndexExpr i0, IndexExpr i1) {
    //            Layout layout = buffer.layout().flatten();
    //            long stride0 = layout.stride().flatAt(0) * buffer.dataType().byteSize();
    //            long stride1 = layout.stride().flatAt(1) * buffer.dataType().byteSize();
    //            IndexExpr term0 = IndexBinary.multiply(i0, IndexConst.of(stride0));
    //            IndexExpr term1 = IndexBinary.multiply(i1, IndexConst.of(stride1));
    //            return IndexBinary.add(term0, term1);
    //        }
    //    }

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
        MemoryView<?> view = iterations.materialize();
        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>) Environment.current().nativeBackend().memoryDomain();
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
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

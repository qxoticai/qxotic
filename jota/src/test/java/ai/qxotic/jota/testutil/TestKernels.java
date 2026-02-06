package ai.qxotic.jota.testutil;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;

public final class TestKernels {

    private static final String PPM_PROPERTY = "jota.test.ppm";

    private TestKernels() {}

    public static float gelu(float x) {
        float cubic = x * x * x;
        float inner = (x + 0.044715f * cubic) * 0.7978845608f;
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    public static Tensor mandelbrotTensor(int width, int height, int iterations) {
        Shape shape = Shape.of(height, width);
        float xMin = -2.5f;
        float xMax = 1.0f;
        float yMin = -1.25f;
        float yMax = 1.25f;
        float xStep = (xMax - xMin) / (width - 1);
        float yStep = (yMax - yMin) / (height - 1);

        Tensor xCoords =
                Tensor.iota(width, DataType.FP32)
                        .multiply(xStep)
                        .add(xMin)
                        .view(Shape.of(1, width));
        Tensor yCoords =
                Tensor.iota(height, DataType.FP32)
                        .multiply(yStep)
                        .add(yMin)
                        .view(Shape.of(height, 1));
        Tensor cReal = xCoords.broadcast(shape);
        Tensor cImag = yCoords.broadcast(shape);

        Tensor zReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zImag = Tensor.zeros(DataType.FP32, shape);
        Tensor iters = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.BOOL, shape);

        Tensor four = Tensor.scalar(4.0f);
        for (int i = 0; i < iterations; i++) {
            Tensor zReal2 = zReal.square();
            Tensor zImag2 = zImag.square();
            Tensor zRealNew = zReal2.subtract(zImag2).add(cReal);
            Tensor zImagNew = zReal.multiply(zImag).multiply(2.0f).add(cImag);

            Tensor magnitude2 = zRealNew.square().add(zImagNew.square());
            Tensor hasEscaped = magnitude2.greaterThan(four);
            Tensor notYetEscaped = escaped.logicalNot();
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            Tensor iterValue = Tensor.full((float) i, DataType.FP32, shape);
            iters = Tensor.where(justEscaped, iterValue, iters);
            escaped = escaped.logicalOr(hasEscaped);
            zReal = zRealNew;
            zImag = zImagNew;
        }

        Tensor finalIter = Tensor.full((float) (iterations - 1), DataType.FP32, shape);
        return Tensor.where(escaped, iters, finalIter);
    }

    public static float mandelbrotIter(
            int row, int col, int width, int height, int iterations) {
        float xMin = -2.5f;
        float xMax = 1.0f;
        float yMin = -1.25f;
        float yMax = 1.25f;
        float xStep = (xMax - xMin) / (width - 1);
        float yStep = (yMax - yMin) / (height - 1);
        float cReal = xMin + col * xStep;
        float cImag = yMin + row * yStep;
        float zReal = 0.0f;
        float zImag = 0.0f;
        for (int i = 0; i < iterations; i++) {
            float zr2 = zReal * zReal;
            float zi2 = zImag * zImag;
            float zRealNew = zr2 - zi2 + cReal;
            float zImagNew = 2.0f * zReal * zImag + cImag;
            float mag2 = zRealNew * zRealNew + zImagNew * zImagNew;
            if (mag2 > 4.0f) {
                return i;
            }
            zReal = zRealNew;
            zImag = zImagNew;
        }
        return iterations - 1;
    }

    public static Tensor phoenixTensor(int width, int height, int iterations) {
        Shape shape = Shape.of(height, width);
        float xMin = -2.0f;
        float xMax = 2.0f;
        float yMin = -1.5f;
        float yMax = 1.5f;
        float xStep = (xMax - xMin) / (width - 1);
        float yStep = (yMax - yMin) / (height - 1);

        Tensor xCoords =
                Tensor.iota(width, DataType.FP32)
                        .multiply(xStep)
                        .add(xMin)
                        .view(Shape.of(1, width));
        Tensor yCoords =
                Tensor.iota(height, DataType.FP32)
                        .multiply(yStep)
                        .add(yMin)
                        .view(Shape.of(height, 1));
        Tensor cReal = xCoords.broadcast(shape);
        Tensor cImag = yCoords.broadcast(shape);

        Tensor zReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zImag = Tensor.zeros(DataType.FP32, shape);
        Tensor zPrevReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zPrevImag = Tensor.zeros(DataType.FP32, shape);
        Tensor iters = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.BOOL, shape);

        Tensor four = Tensor.scalar(4.0f);
        Tensor pReal = Tensor.scalar(-0.5f);
        Tensor pImag = Tensor.scalar(0.0f);

        for (int i = 0; i < iterations; i++) {
            Tensor zReal2 = zReal.square();
            Tensor zImag2 = zImag.square();
            Tensor prodReal = pReal.multiply(zPrevReal).subtract(pImag.multiply(zPrevImag));
            Tensor prodImag = pReal.multiply(zPrevImag).add(pImag.multiply(zPrevReal));
            Tensor zRealNew = zReal2.subtract(zImag2).add(cReal).add(prodReal);
            Tensor zImagNew = zReal.multiply(zImag).multiply(2.0f).add(cImag).add(prodImag);

            Tensor magnitude2 = zRealNew.square().add(zImagNew.square());
            Tensor hasEscaped = magnitude2.greaterThan(four);
            Tensor notYetEscaped = escaped.logicalNot();
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            Tensor iterValue = Tensor.full((float) i, DataType.FP32, shape);
            iters = Tensor.where(justEscaped, iterValue, iters);
            escaped = escaped.logicalOr(hasEscaped);

            zPrevReal = zReal;
            zPrevImag = zImag;
            zReal = zRealNew;
            zImag = zImagNew;
        }

        Tensor finalIter = Tensor.full((float) (iterations - 1), DataType.FP32, shape);
        return Tensor.where(escaped, iters, finalIter);
    }

    public static float phoenixIter(
            int row, int col, int width, int height, int iterations) {
        float xMin = -2.0f;
        float xMax = 2.0f;
        float yMin = -1.5f;
        float yMax = 1.5f;
        float xStep = (xMax - xMin) / (width - 1);
        float yStep = (yMax - yMin) / (height - 1);
        float cReal = xMin + col * xStep;
        float cImag = yMin + row * yStep;

        float zReal = 0.0f;
        float zImag = 0.0f;
        float zPrevReal = 0.0f;
        float zPrevImag = 0.0f;
        float pReal = -0.5f;
        float pImag = 0.0f;

        for (int i = 0; i < iterations; i++) {
            float zr2 = zReal * zReal;
            float zi2 = zImag * zImag;
            float prodReal = pReal * zPrevReal - pImag * zPrevImag;
            float prodImag = pReal * zPrevImag + pImag * zPrevReal;
            float zRealNew = zr2 - zi2 + cReal + prodReal;
            float zImagNew = 2.0f * zReal * zImag + cImag + prodImag;
            float mag2 = zRealNew * zRealNew + zImagNew * zImagNew;
            if (mag2 > 4.0f) {
                return i;
            }
            zPrevReal = zReal;
            zPrevImag = zImag;
            zReal = zRealNew;
            zImag = zImagNew;
        }
        return iterations - 1;
    }

    public static void writeMandelbrotPpm(
            MemoryContext<MemorySegment> context,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations) {
        if (!Boolean.getBoolean(PPM_PROPERTY)) {
            return;
        }
        writePpm(context, view, path, width, height, iterations, false);
    }

    public static void writePhoenixPpm(
            MemoryContext<MemorySegment> context,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations) {
        if (!Boolean.getBoolean(PPM_PROPERTY)) {
            return;
        }
        writePpm(context, view, path, width, height, iterations, true);
    }

    private static void writePpm(
            MemoryContext<MemorySegment> context,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations,
            boolean phoenixPalette) {
        try {
            Files.createDirectories(path.getParent());
            try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(path))) {
                writeAscii(stream, "P3\n");
                writeAscii(stream, width + " " + height + "\n");
                writeAscii(stream, "255\n");

                MemoryAccess<MemorySegment> access = context.memoryAccess();
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        long idx = (long) h * width + w;
                        long offset = Indexing.linearToOffset(view, idx);
                        float iter = access.readFloat(view.memory(), offset);
                        int[] rgb = phoenixPalette
                                ? phoenixColor(iter, iterations)
                                : mandelbrotColor(iter, iterations);
                        writeAscii(stream, rgb[0] + " " + rgb[1] + " " + rgb[2]);
                        if (w < width - 1) {
                            writeAscii(stream, " ");
                        }
                    }
                    writeAscii(stream, "\n");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write PPM to " + path, e);
        }
    }

    private static int[] mandelbrotColor(float iter, int iterations) {
        if (iter >= iterations - 1) {
            return new int[] {0, 0, 0};
        }
        double t = iter / iterations;
        int r = (int) (9 * (1 - t) * t * t * t * 255);
        int g = (int) (15 * (1 - t) * (1 - t) * t * t * 255);
        int b = (int) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        return new int[] {r, g, b};
    }

    private static int[] phoenixColor(float iter, int iterations) {
        if (iter >= iterations - 1) {
            return new int[] {6, 6, 12};
        }
        double t = iter / iterations;
        double gamma = Math.pow(t, 0.55);
        int r = (int) (255 * Math.min(1.0, 0.8 * Math.sin(6.2831 * gamma) + 0.2));
        int g = (int) (255 * Math.min(1.0, 0.7 * Math.sin(6.2831 * (gamma + 0.33)) + 0.3));
        int b = (int) (255 * Math.min(1.0, 0.9 * Math.sin(6.2831 * (gamma + 0.67)) + 0.1));
        if (r < 0) {
            r = 0;
        }
        if (g < 0) {
            g = 0;
        }
        if (b < 0) {
            b = 0;
        }
        return new int[] {r, g, b};
    }

    private static void writeAscii(OutputStream stream, String value) throws IOException {
        stream.write(value.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
    }
}

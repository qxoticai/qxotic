package com.qxotic.jota.testutil;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class TestKernels {

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
            iters = justEscaped.where(iterValue, iters);
            escaped = escaped.logicalOr(hasEscaped);
            zReal = zRealNew;
            zImag = zImagNew;
        }

        Tensor finalIter = Tensor.full((float) (iterations - 1), DataType.FP32, shape);
        return escaped.where(iters, finalIter);
    }

    public static float mandelbrotIter(int row, int col, int width, int height, int iterations) {
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

    public static void writeMandelbrotPpm(
            MemoryDomain<MemorySegment> domain,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations) {
        writePpm(domain, view, path, width, height, iterations);
    }

    public static void writePhoenixPpm(
            MemoryDomain<MemorySegment> domain,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations) {
        writePpm(domain, view, path, width, height, iterations);
    }

    private static void writePpm(
            MemoryDomain<MemorySegment> domain,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height,
            int iterations) {
        try {
            Files.createDirectories(path.getParent());
            try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(path))) {
                writeAscii(stream, "P6\n");
                writeAscii(stream, width + " " + height + "\n");
                writeAscii(stream, "255\n");

                MemoryAccess<MemorySegment> access = domain.directAccess();
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        long idx = (long) h * width + w;
                        long offset = Indexing.linearToOffset(view, idx);
                        float iter = access.readFloat(view.memory(), offset);
                        int[] rgb = mandelbrotColor(iter, iterations);
                        stream.write(rgb[0]);
                        stream.write(rgb[1]);
                        stream.write(rgb[2]);
                    }
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

    private static void writeAscii(OutputStream stream, String value) throws IOException {
        stream.write(value.getBytes(StandardCharsets.US_ASCII));
    }
}

package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.LazyComputation;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import org.junit.jupiter.api.Test;

class LIRInterpreterTraceTest {

    private static final int MANDEL_WIDTH = 1920;
    private static final int MANDEL_HEIGHT = 1080;
    private static final int MANDEL_ITER = 100;
    private static final float X_MIN = -2.5f;
    private static final float X_MAX = 1.0f;
    private static final float Y_MIN = -1.25f;
    private static final float Y_MAX = 1.25f;

    @Test
    void interpretsTracedGelu() {
        MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();
        MemoryView<MemorySegment> input =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.flat(4));
        Tensor inputTensor = Tensor.of(input);

        Tensor traced = Tracer.trace(inputTensor, Tensor::gelu);
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(context, lirGraph.outputs().getFirst());

        new LIRInterpreter()
                .execute(lirGraph, List.of(input), List.of(), List.of(output), context);

        assertEquals(gelu(0.0f), readFloat(context, output, 0), 1e-4f);
        assertEquals(gelu(1.0f), readFloat(context, output, 1), 1e-4f);
        assertEquals(gelu(2.0f), readFloat(context, output, 2), 1e-4f);
        assertEquals(gelu(3.0f), readFloat(context, output, 3), 1e-4f);
    }

    @Test
    void interpretsTracedMandelbrot() {
        MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();
        Tensor traced = Tracer.trace(List.of(), inputs -> computeMandelbrotTensor());
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(context, lirGraph.outputs().getFirst());

        new LIRInterpreter().execute(lirGraph, List.of(), List.of(), List.of(output), context);

        writeMandelbrotPpm(context, output, Path.of("target", "mandelbrot-lir.ppm"));

        assertEquals(
                mandelbrotIter(0, 0),
                readFloat(context, output, 0),
                1e-3f);
        assertEquals(
                mandelbrotIter(MANDEL_HEIGHT / 2, MANDEL_WIDTH / 2),
                readFloat(
                        context,
                        output,
                        (MANDEL_HEIGHT / 2L) * MANDEL_WIDTH + (MANDEL_WIDTH / 2L)),
                1e-3f);
        assertEquals(
                mandelbrotIter(MANDEL_HEIGHT - 1, MANDEL_WIDTH - 1),
                readFloat(
                        context,
                        output,
                        (long) (MANDEL_HEIGHT - 1) * MANDEL_WIDTH + (MANDEL_WIDTH - 1)),
                1e-3f);
    }

    private TIRGraph extractGraph(Tensor traced) {
        LazyComputation computation = traced.computation().orElseThrow();
        Object graph = computation.attributes().get("graph");
        if (!(graph instanceof TIRGraph tirGraph)) {
            throw new IllegalStateException("Expected TIRGraph, got: " + graph);
        }
        return tirGraph;
    }

    private MemoryView<MemorySegment> allocateOutput(
            MemoryContext<MemorySegment> context, BufferRef buffer) {
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(buffer.dataType(), buffer.layout().shape());
        return MemoryView.of(memory, buffer.dataType(), buffer.layout());
    }

    private float readFloat(
            MemoryContext<MemorySegment> context, MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return access.readFloat(typedView.memory(), offset);
    }

    private float gelu(float x) {
        float cubic = x * x * x;
        float inner = (x + 0.044715f * cubic) * 0.7978845608f;
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    private Tensor computeMandelbrotTensor() {
        Shape shape = Shape.of(MANDEL_HEIGHT, MANDEL_WIDTH);

        float xStep = (X_MAX - X_MIN) / (MANDEL_WIDTH - 1);
        float yStep = (Y_MAX - Y_MIN) / (MANDEL_HEIGHT - 1);
        Tensor xCoords =
                Tensor.iota(MANDEL_WIDTH, DataType.FP32)
                        .multiply(xStep)
                        .add(X_MIN)
                        .view(Shape.of(1, MANDEL_WIDTH));
        Tensor yCoords =
                Tensor.iota(MANDEL_HEIGHT, DataType.FP32)
                        .multiply(yStep)
                        .add(Y_MIN)
                        .view(Shape.of(MANDEL_HEIGHT, 1));
        Tensor cReal = xCoords.broadcast(shape);
        Tensor cImag = yCoords.broadcast(shape);

        Tensor zReal = Tensor.zeros(DataType.FP32, shape);
        Tensor zImag = Tensor.zeros(DataType.FP32, shape);
        Tensor iterations = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.BOOL, shape);

        Tensor four = Tensor.scalar(4.0f);
        for (int i = 0; i < MANDEL_ITER; i++) {
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

        Tensor finalIter = Tensor.full((float) (MANDEL_ITER - 1), DataType.FP32, shape);
        return Tensor.where(escaped, iterations, finalIter);
    }

    private float mandelbrotIter(int row, int col) {
        float xStep = (X_MAX - X_MIN) / (MANDEL_WIDTH - 1);
        float yStep = (Y_MAX - Y_MIN) / (MANDEL_HEIGHT - 1);
        float cReal = X_MIN + col * xStep;
        float cImag = Y_MIN + row * yStep;
        float zReal = 0.0f;
        float zImag = 0.0f;
        for (int i = 0; i < MANDEL_ITER; i++) {
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
        return MANDEL_ITER - 1;
    }

    private void writeMandelbrotPpm(
            MemoryContext<MemorySegment> context, MemoryView<?> view, Path path) {
        try {
            Files.createDirectories(path.getParent());
            try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(path))) {
                writeAscii(stream, "P3\n");
                writeAscii(stream, MANDEL_WIDTH + " " + MANDEL_HEIGHT + "\n");
                writeAscii(stream, "255\n");

                MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
                MemoryAccess<MemorySegment> access = context.memoryAccess();
                for (int h = 0; h < MANDEL_HEIGHT; h++) {
                    for (int w = 0; w < MANDEL_WIDTH; w++) {
                        long idx = (long) h * MANDEL_WIDTH + w;
                        long offset = Indexing.linearToOffset(typedView, idx);
                        float iter = access.readFloat(typedView.memory(), offset);
                        int[] rgb = mandelbrotColor(iter);
                        writeAscii(stream, rgb[0] + " " + rgb[1] + " " + rgb[2]);
                        if (w < MANDEL_WIDTH - 1) {
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

    private int[] mandelbrotColor(float iter) {
        if (iter >= MANDEL_ITER - 1) {
            return new int[] {0, 0, 0};
        }
        double t = iter / MANDEL_ITER;
        int r = (int) (9 * (1 - t) * t * t * t * 255);
        int g = (int) (15 * (1 - t) * (1 - t) * t * t * 255);
        int b = (int) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        return new int[] {r, g, b};
    }

    private void writeAscii(OutputStream stream, String value) throws IOException {
        stream.write(value.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
    }
}

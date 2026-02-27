package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class NewtonFractalTest {

    private static final int WIDTH = 320;
    private static final int HEIGHT = 240;
    private static final int ITERATIONS = 40;

    private static final float X_MIN = -1.5f;
    private static final float X_MAX = 1.5f;
    private static final float Y_MIN = -1.2f;
    private static final float Y_MAX = 1.2f;

    @Test
    void generatesNewtonPpmOnCurrentBackend() {
        Device backend = Environment.current().defaultDevice();
        MemoryView<?> output = Tracer.trace(() -> newtonTensor(WIDTH, HEIGHT, ITERATIONS)).materialize();

        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView = toHost(output);
        MemoryAccess<MemorySegment> access = hostDomain.directAccess();

        Path path = Path.of("target", "newton-" + backend.leafName() + "-lir.ppm");
        writeNewtonPpm(access, hostView, path, WIDTH, HEIGHT);

        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            float value = access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, i));
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        assertTrue(min >= 0.0f);
        assertTrue(max < 3.0f);
    }

    private static Tensor newtonTensor(int width, int height, int iterations) {
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

        Tensor iterFrac = Tensor.zeros(DataType.FP32, shape);
        Tensor solved = Tensor.zeros(DataType.BOOL, shape);
        Tensor threshold = Tensor.scalar(1e-8f);

        for (int i = 0; i < iterations; i++) {
            Tensor zr2 = zReal.square();
            Tensor zi2 = zImag.square();
            Tensor zr3 = zr2.multiply(zReal).subtract(zReal.multiply(zi2).multiply(3.0f));
            Tensor zi3 = zr2.multiply(zImag).multiply(3.0f).subtract(zi2.multiply(zImag));

            Tensor fReal = zr3.subtract(1.0f);
            Tensor fImag = zi3;

            Tensor fpReal = zr2.subtract(zi2).multiply(3.0f);
            Tensor fpImag = zReal.multiply(zImag).multiply(6.0f);
            Tensor denom = fpReal.square().add(fpImag.square()).add(1e-12f);

            Tensor deltaReal = fReal.multiply(fpReal).add(fImag.multiply(fpImag)).divide(denom);
            Tensor deltaImag = fImag.multiply(fpReal).subtract(fReal.multiply(fpImag)).divide(denom);

            Tensor nextReal = zReal.subtract(deltaReal);
            Tensor nextImag = zImag.subtract(deltaImag);
            Tensor delta2 = deltaReal.square().add(deltaImag.square());

            Tensor justSolved = delta2.lessThan(threshold).logicalAnd(solved.logicalNot());
            Tensor frac = Tensor.full((float) i / iterations, DataType.FP32, shape);
            iterFrac = justSolved.where(frac, iterFrac);
            solved = solved.logicalOr(delta2.lessThan(threshold));
            zReal = nextReal;
            zImag = nextImag;
        }

        float rt3o2 = (float) (Math.sqrt(3.0) / 2.0);
        Tensor d0 = zReal.subtract(1.0f).square().add(zImag.square());
        Tensor d1 = zReal.add(0.5f).square().add(zImag.subtract(rt3o2).square());
        Tensor d2 = zReal.add(0.5f).square().add(zImag.add(rt3o2).square());

        Tensor root0 = d0.lessThan(d1).logicalAnd(d0.lessThan(d2));
        Tensor root1 = d1.lessThan(d2);
        Tensor rootId =
                root0.where(
                        Tensor.scalar(0.0f),
                        root1.where(Tensor.scalar(1.0f), Tensor.scalar(2.0f)));
        Tensor unresolved = solved.logicalNot();
        Tensor unresolvedCode = Tensor.full(2.95f, DataType.FP32, shape);
        return unresolved.where(unresolvedCode, rootId.add(iterFrac.multiply(0.95f)));
    }

    private static void writeNewtonPpm(
            MemoryAccess<MemorySegment> access,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height) {
        try {
            Files.createDirectories(path.getParent());
            try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(path))) {
                writeAscii(stream, "P6\n");
                writeAscii(stream, width + " " + height + "\n");
                writeAscii(stream, "255\n");
                for (int i = 0; i < width * height; i++) {
                    float code = access.readFloat(view.memory(), Indexing.linearToOffset(view, i));
                    int[] rgb = newtonColor(code);
                    stream.write(rgb[0]);
                    stream.write(rgb[1]);
                    stream.write(rgb[2]);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write Newton PPM to " + path, e);
        }
    }

    private static int[] newtonColor(float code) {
        int root = Math.max(0, Math.min(2, (int) code));
        float t = Math.max(0.0f, Math.min(1.0f, code - root));
        int[][] base = {
            {255, 100, 70},
            {80, 210, 120},
            {90, 150, 255}
        };
        int r = (int) Math.round(base[root][0] * (1.0f - 0.7f * t));
        int g = (int) Math.round(base[root][1] * (1.0f - 0.7f * t));
        int b = (int) Math.round(base[root][2] * (1.0f - 0.7f * t));
        return new int[] {r, g, b};
    }

    private static void writeAscii(OutputStream stream, String value) throws IOException {
        stream.write(value.getBytes(StandardCharsets.US_ASCII));
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        return (MemoryView<MemorySegment>) Tensor.of(view).to(Device.PANAMA).materialize();
    }
}

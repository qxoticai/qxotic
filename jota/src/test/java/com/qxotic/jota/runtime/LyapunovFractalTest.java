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
class LyapunovFractalTest {

    private static final boolean HIGH_QUALITY = Boolean.getBoolean("jota.fractal.highQuality");
    private static final int WIDTH = 320;
    private static final int HEIGHT = 240;
    private static final int WARMUP = HIGH_QUALITY ? 64 : 24;
    private static final int SAMPLES = HIGH_QUALITY ? 384 : 128;
    private static final int TRACE_CHUNK = HIGH_QUALITY ? 32 : 24;
    private static final String SEQUENCE = "AABAB";

    private static final float R_MIN = 2.4f;
    private static final float R_MAX = 4.0f;

    @Test
    void generatesLyapunovPpmOnCurrentBackend() {
        Device backend = Environment.current().defaultDevice();
        MemoryView<?> output =
                lyapunovTensor(WIDTH, HEIGHT, WARMUP, SAMPLES, TRACE_CHUNK).materialize();

        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> hostView = toHost(output);
        MemoryAccess<MemorySegment> access = hostDomain.directAccess();

        Path path = Path.of("target", "lyapunov-" + backend.leafName() + "-lir.ppm");
        writeLyapunovPpm(access, hostView, path, WIDTH, HEIGHT);

        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            float value = access.readFloat(hostView.memory(), Indexing.linearToOffset(hostView, i));
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        assertTrue(min < -0.01f);
        assertTrue(max > 0.01f);
    }

    private static Tensor lyapunovTensor(
            int width, int height, int warmup, int samples, int chunkSize) {
        Shape shape = Shape.of(height, width);
        float aStep = (R_MAX - R_MIN) / (width - 1);
        float bStep = (R_MAX - R_MIN) / (height - 1);

        Tensor rA =
                Tensor.iota(width, DataType.FP32)
                        .multiply(aStep)
                        .add(R_MIN)
                        .view(Shape.of(1, width))
                        .broadcast(shape);
        Tensor rB =
                Tensor.iota(height, DataType.FP32)
                        .multiply(bStep)
                        .add(R_MIN)
                        .view(Shape.of(height, 1))
                        .broadcast(shape);

        Tensor x = Tensor.full(0.5f, DataType.FP32, shape).add(rA.subtract(R_MIN).multiply(1e-3f));
        Tensor sum = Tensor.zeros(DataType.FP32, shape);
        Tensor one = Tensor.scalar(1.0f);
        Tensor two = Tensor.scalar(2.0f);
        Tensor eps = Tensor.scalar(1e-6f);

        int total = warmup + samples;
        int start = 0;
        while (start < total) {
            int endExclusive = Math.min(start + chunkSize, total);
            for (int i = start; i < endExclusive; i++) {
                Tensor r = (SEQUENCE.charAt(i % SEQUENCE.length()) == 'A') ? rA : rB;
                x = r.multiply(x).multiply(one.subtract(x));
                if (i >= warmup) {
                    Tensor derivAbs = r.multiply(one.subtract(x.multiply(two)).abs());
                    Tensor safe = derivAbs.lessThan(eps).where(eps, derivAbs);
                    sum = sum.add(safe.log());
                }
            }
            x = Tensor.of(x.materialize());
            sum = Tensor.of(sum.materialize());
            start = endExclusive;
        }
        return sum.divide((float) samples);
    }

    private static void writeLyapunovPpm(
            MemoryAccess<MemorySegment> access,
            MemoryView<MemorySegment> view,
            Path path,
            int width,
            int height) {
        float maxAbs = 1e-6f;
        for (int i = 0; i < width * height; i++) {
            float value = access.readFloat(view.memory(), Indexing.linearToOffset(view, i));
            maxAbs = Math.max(maxAbs, Math.abs(value));
        }

        try {
            Files.createDirectories(path.getParent());
            try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(path))) {
                writeAscii(stream, "P6\n");
                writeAscii(stream, width + " " + height + "\n");
                writeAscii(stream, "255\n");
                for (int i = 0; i < width * height; i++) {
                    float value = access.readFloat(view.memory(), Indexing.linearToOffset(view, i));
                    int[] rgb = lyapunovColor(value, maxAbs);
                    stream.write(rgb[0]);
                    stream.write(rgb[1]);
                    stream.write(rgb[2]);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write Lyapunov PPM to " + path, e);
        }
    }

    private static int[] lyapunovColor(float value, float maxAbs) {
        float normalized = (float) Math.tanh(1.75f * (value / maxAbs));
        if (normalized < 0.0f) {
            float t = -normalized;
            int r = (int) Math.round(8 + 70 * t);
            int g = (int) Math.round(35 + 210 * t);
            int b = (int) Math.round(80 + 170 * t);
            return new int[] {r, g, b};
        }
        float t = normalized;
        int r = (int) Math.round(75 + 180 * t);
        int g = (int) Math.round(35 + 85 * t);
        int b = (int) Math.round(18 + 45 * t);
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

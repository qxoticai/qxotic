package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class SpiralGalaxyDemoTest {

    private static final int WIDTH = 256;
    private static final int HEIGHT = 256;
    private static final float FREQUENCY = 6.0f;
    private static final float TWIST = 8.0f;
    private static final float FALLOFF = 2.5f;

    @SuppressWarnings("unchecked")
    private static final MemoryContext<MemorySegment> CONTEXT =
            (MemoryContext<MemorySegment>) Environment.current().nativeBackend().memoryContext();

    @Test
    void rendersSpiralGalaxyField() throws IOException {
        Tensor x = Tensor.arange(WIDTH, DataType.FP32).multiply(2.0f / (WIDTH - 1)).add(-1.0f);
        Tensor y = Tensor.arange(HEIGHT, DataType.FP32).multiply(2.0f / (HEIGHT - 1)).add(-1.0f);

        Tensor xGrid = x.view(Shape.of(1, WIDTH)).broadcast(Shape.of(HEIGHT, WIDTH));
        Tensor yGrid = y.view(Shape.of(HEIGHT, 1)).broadcast(Shape.of(HEIGHT, WIDTH));

        Tensor radius = xGrid.square().add(yGrid.square()).sqrt();
        Tensor theta = radius.multiply(TWIST);

        Tensor xRot = xGrid.multiply(theta.cos()).subtract(yGrid.multiply(theta.sin()));
        Tensor arms = xRot.multiply(FREQUENCY).sin().abs();
        Tensor fade = radius.multiply(FALLOFF).negate().exp();
        Tensor image = arms.multiply(fade);

        MemoryView<?> output = image.materialize();

        // writePPM(output, "target/spiral-galaxy.ppm");

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(HEIGHT, WIDTH), output.shape());

        assertPixel(output, 0, 0);
        assertPixel(output, HEIGHT / 2, WIDTH / 2);
        assertPixel(output, HEIGHT / 2, 0);
        assertPixel(output, 0, WIDTH / 2);
        assertPixel(output, HEIGHT / 4, WIDTH / 3);
    }

    private static void assertPixel(MemoryView<?> view, int h, int w) {
        int linearIndex = h * WIDTH + w;
        float actual = readFloat(view, linearIndex);
        float expected = expectedIntensity(h, w);
        assertEquals(expected, actual, 0.001f);
        assertTrue(actual >= -0.0001f);
        assertTrue(actual <= 1.0001f);
    }

    private static float expectedIntensity(int h, int w) {
        float x = -1.0f + 2.0f * w / (WIDTH - 1);
        float y = -1.0f + 2.0f * h / (HEIGHT - 1);
        float r = (float) Math.sqrt(x * x + y * y);
        float theta = r * TWIST;
        float xRot = x * (float) Math.cos(theta) - y * (float) Math.sin(theta);
        float arms = Math.abs((float) Math.sin(xRot * FREQUENCY));
        float fade = (float) Math.exp(-r * FALLOFF);
        return arms * fade;
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.memoryAccess().readFloat(typedView.memory(), offset);
    }

    private static void writePPM(MemoryView<?> view, String filename) throws IOException {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = CONTEXT.memoryAccess();
        try (PrintStream out =
                new PrintStream(new BufferedOutputStream(new FileOutputStream(filename)))) {
            out.println("P3");
            out.println(WIDTH + " " + HEIGHT);
            out.println("255");
            for (int h = 0; h < HEIGHT; h++) {
                for (int w = 0; w < WIDTH; w++) {
                    int idx = h * WIDTH + w;
                    long offset = Indexing.linearToOffset(typedView, idx);
                    float value = access.readFloat(typedView.memory(), offset);
                    int shade = clampToByte(value * 255.0f);
                    out.print(shade + " " + shade + " " + shade);
                    if (w < WIDTH - 1) {
                        out.print(" ");
                    }
                }
                out.println();
            }
        }
    }

    private static int clampToByte(float value) {
        if (value <= 0.0f) {
            return 0;
        }
        if (value >= 255.0f) {
            return 255;
        }
        return Math.round(value);
    }
}

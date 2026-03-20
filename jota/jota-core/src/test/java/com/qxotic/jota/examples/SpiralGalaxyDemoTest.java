package com.qxotic.jota.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.PpmWriter;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class SpiralGalaxyDemoTest {

    private static final int WIDTH = 256;
    private static final int HEIGHT = 256;
    private static final float FREQUENCY = 6.0f;
    private static final float TWIST = 8.0f;
    private static final float FALLOFF = 2.5f;

    @Test
    void rendersSpiralGalaxyField() throws IOException {
        Tensor x = Tensor.iota(WIDTH, DataType.FP32).multiply(2.0f / (WIDTH - 1)).add(-1.0f);
        Tensor y = Tensor.iota(HEIGHT, DataType.FP32).multiply(2.0f / (HEIGHT - 1)).add(-1.0f);

        Tensor xGrid = x.view(Shape.of(1, WIDTH)).broadcast(Shape.of(HEIGHT, WIDTH));
        Tensor yGrid = y.view(Shape.of(HEIGHT, 1)).broadcast(Shape.of(HEIGHT, WIDTH));

        Tensor radius = xGrid.square().add(yGrid.square()).sqrt();
        Tensor theta = radius.multiply(TWIST);

        Tensor xRot = xGrid.multiply(theta.cos()).subtract(yGrid.multiply(theta.sin()));
        Tensor arms = xRot.multiply(FREQUENCY).sin().abs();
        Tensor fade = radius.multiply(FALLOFF).negate().exp();
        Tensor image = arms.multiply(fade);

        MemoryView<?> output = image.materialize();

        writePPM(output, "target/spiral-galaxy.ppm");

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(HEIGHT, WIDTH), output.shape());

        assertPixel(image, 0, 0);
        assertPixel(image, HEIGHT / 2, WIDTH / 2);
        assertPixel(image, HEIGHT / 2, 0);
        assertPixel(image, 0, WIDTH / 2);
        assertPixel(image, HEIGHT / 4, WIDTH / 3);
    }

    private static void assertPixel(Tensor tensor, int h, int w) {
        int linearIndex = h * WIDTH + w;
        float actual = readFloat(tensor, linearIndex);
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

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }

    private static void writePPM(MemoryView<?> view, String filename) throws IOException {
        MemoryView<?> hostView =
                Tensor.of(view).to(Environment.current().nativeRuntime().device()).materialize();
        MemorySegment segment = (MemorySegment) hostView.memory().base();
        byte[] rgb = new byte[WIDTH * HEIGHT * 3];
        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                int idx = h * WIDTH + w;
                long offset = Indexing.linearToOffset(hostView, idx);
                float value = segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
                byte shade = (byte) clampToByte(value * 255.0f);
                rgb[idx * 3] = shade;
                rgb[idx * 3 + 1] = shade;
                rgb[idx * 3 + 2] = shade;
            }
        }
        PpmWriter.write(Path.of(filename), WIDTH, HEIGHT, rgb);
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

package com.qxotic.jota.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class NBodySnapshotTest {

    private static final int BODIES = 32;
    private static final int STEPS = 10;
    private static final float DT = 0.01f;
    private static final float G = 0.04f;
    private static final float SOFTENING = 1e-3f;
    private static final float DAMPING = 0.999f;

    private static final int IMAGE_W = 320;
    private static final int IMAGE_H = 240;
    private static final float WORLD_RADIUS = 1.0f;

    @Test
    void nBodySnapshotMatchesReferenceAndWritesPpm() throws IOException {
        InitialState init = initialState(BODIES);
        NBodyState tensorState = runTensorSimulation(init, STEPS);
        NBodyState referenceState = runReferenceSimulation(init, STEPS);

        for (int i = 0; i < BODIES; i++) {
            assertEquals(referenceState.x[i], tensorState.x[i], 2e-3f, "x mismatch at " + i);
            assertEquals(referenceState.y[i], tensorState.y[i], 2e-3f, "y mismatch at " + i);
            assertEquals(referenceState.vx[i], tensorState.vx[i], 3e-3f, "vx mismatch at " + i);
            assertEquals(referenceState.vy[i], tensorState.vy[i], 3e-3f, "vy mismatch at " + i);
        }

        writeSnapshotPpm(
                tensorState,
                init.mass,
                Path.of(
                        "target",
                        "nbody-snapshot-"
                                + Environment.current().defaultDevice().leafName()
                                + ".ppm"));
    }

    private static NBodyState runTensorSimulation(InitialState init, int steps) {
        Shape vecShape = Shape.of(BODIES, 1);
        Shape pairShape = Shape.of(BODIES, BODIES);

        Tensor x = Tensor.of(init.x, vecShape);
        Tensor y = Tensor.of(init.y, vecShape);
        Tensor vx = Tensor.of(init.vx, vecShape);
        Tensor vy = Tensor.of(init.vy, vecShape);
        Tensor mass = Tensor.of(init.mass, vecShape);
        Tensor pairMask = Tensor.of(init.pairMask, pairShape);
        Tensor onesCol = Tensor.ones(vecShape);

        for (int step = 0; step < steps; step++) {
            Tensor xT = x.transpose(0, 1);
            Tensor yT = y.transpose(0, 1);
            Tensor mT = mass.transpose(0, 1);

            Tensor xi = x.broadcast(pairShape);
            Tensor yi = y.broadcast(pairShape);
            Tensor xj = xT.broadcast(pairShape);
            Tensor yj = yT.broadcast(pairShape);

            Tensor dx = xj.subtract(xi);
            Tensor dy = yj.subtract(yi);

            Tensor dist2 = dx.square().add(dy.square()).add(SOFTENING);
            Tensor invDist = dist2.sqrt().reciprocal();
            Tensor invDist3 = invDist.square().multiply(invDist);

            Tensor mj = mT.broadcast(pairShape);
            Tensor accelScale = invDist3.multiply(mj).multiply(pairMask).multiply(G);

            Tensor ax = dx.multiply(accelScale).matmul(onesCol);
            Tensor ay = dy.multiply(accelScale).matmul(onesCol);

            vx = vx.add(ax.multiply(DT)).multiply(DAMPING);
            vy = vy.add(ay.multiply(DT)).multiply(DAMPING);

            x = x.add(vx.multiply(DT));
            y = y.add(vy.multiply(DT));

            x = Tensor.of(x.materialize());
            y = Tensor.of(y.materialize());
            vx = Tensor.of(vx.materialize());
            vy = Tensor.of(vy.materialize());
        }

        float[] outX = new float[BODIES];
        float[] outY = new float[BODIES];
        float[] outVx = new float[BODIES];
        float[] outVy = new float[BODIES];
        for (int i = 0; i < BODIES; i++) {
            outX[i] = TensorTestReads.readFloat(x, i);
            outY[i] = TensorTestReads.readFloat(y, i);
            outVx[i] = TensorTestReads.readFloat(vx, i);
            outVy[i] = TensorTestReads.readFloat(vy, i);
        }
        return new NBodyState(outX, outY, outVx, outVy);
    }

    private static NBodyState runReferenceSimulation(InitialState init, int steps) {
        float[] x = init.x.clone();
        float[] y = init.y.clone();
        float[] vx = init.vx.clone();
        float[] vy = init.vy.clone();
        float[] mass = init.mass;

        float[] ax = new float[BODIES];
        float[] ay = new float[BODIES];

        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < BODIES; i++) {
                float sAx = 0.0f;
                float sAy = 0.0f;
                for (int j = 0; j < BODIES; j++) {
                    if (i == j) {
                        continue;
                    }
                    float dx = x[j] - x[i];
                    float dy = y[j] - y[i];
                    float dist2 = dx * dx + dy * dy + SOFTENING;
                    float inv = (float) (1.0 / Math.sqrt(dist2));
                    float inv3 = inv * inv * inv;
                    float scale = G * mass[j] * inv3;
                    sAx += dx * scale;
                    sAy += dy * scale;
                }
                ax[i] = sAx;
                ay[i] = sAy;
            }

            for (int i = 0; i < BODIES; i++) {
                vx[i] = (vx[i] + ax[i] * DT) * DAMPING;
                vy[i] = (vy[i] + ay[i] * DT) * DAMPING;
                x[i] += vx[i] * DT;
                y[i] += vy[i] * DT;
            }
        }
        return new NBodyState(x, y, vx, vy);
    }

    private static InitialState initialState(int count) {
        float[] x = new float[count];
        float[] y = new float[count];
        float[] vx = new float[count];
        float[] vy = new float[count];
        float[] mass = new float[count];
        float[] pairMask = new float[count * count];

        for (int i = 0; i < count; i++) {
            float angle = (float) (2.0 * Math.PI * i / count);
            float radius = 0.4f + 0.1f * (float) Math.sin(i * 0.37f);
            float px = radius * (float) Math.cos(angle);
            float py = radius * (float) Math.sin(angle);

            x[i] = px;
            y[i] = py;
            vx[i] = -py * 0.2f;
            vy[i] = px * 0.2f;
            mass[i] = 0.8f + 0.4f * ((i % 7) / 6.0f);

            int row = i * count;
            for (int j = 0; j < count; j++) {
                pairMask[row + j] = i == j ? 0.0f : 1.0f;
            }
        }
        return new InitialState(x, y, vx, vy, mass, pairMask);
    }

    private static void writeSnapshotPpm(NBodyState state, float[] mass, Path path)
            throws IOException {
        int[] rgb = new int[IMAGE_W * IMAGE_H * 3];

        for (int i = 0; i < state.x.length; i++) {
            int px = worldToPixelX(state.x[i]);
            int py = worldToPixelY(state.y[i]);
            int radius = Math.max(1, Math.min(3, 1 + Math.round((mass[i] - 0.8f) * 4.0f)));
            int red = clamp255(180 + Math.round(60.0f * mass[i]));
            int green = clamp255(160 + Math.round(45.0f * (1.5f - mass[i])));
            int blue = 245;
            drawDisc(rgb, px, py, radius, red, green, blue);
        }

        try (PrintStream out =
                new PrintStream(new BufferedOutputStream(new FileOutputStream(path.toFile())))) {
            out.println("P3");
            out.println(IMAGE_W + " " + IMAGE_H);
            out.println("255");
            for (int y = 0; y < IMAGE_H; y++) {
                for (int x = 0; x < IMAGE_W; x++) {
                    int idx = (y * IMAGE_W + x) * 3;
                    out.print(rgb[idx]);
                    out.print(' ');
                    out.print(rgb[idx + 1]);
                    out.print(' ');
                    out.print(rgb[idx + 2]);
                    if (x < IMAGE_W - 1) {
                        out.print(' ');
                    }
                }
                out.println();
            }
        }
    }

    private static int worldToPixelX(float x) {
        float normalized = (x / WORLD_RADIUS + 1.0f) * 0.5f;
        return clampCoord(Math.round(normalized * (IMAGE_W - 1)), IMAGE_W);
    }

    private static int worldToPixelY(float y) {
        float normalized = (y / WORLD_RADIUS + 1.0f) * 0.5f;
        return clampCoord(Math.round(normalized * (IMAGE_H - 1)), IMAGE_H);
    }

    private static void drawDisc(int[] rgb, int cx, int cy, int radius, int r, int g, int b) {
        int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy > r2) {
                    continue;
                }
                int px = cx + dx;
                int py = cy + dy;
                if (px < 0 || py < 0 || px >= IMAGE_W || py >= IMAGE_H) {
                    continue;
                }
                int idx = (py * IMAGE_W + px) * 3;
                rgb[idx] = r;
                rgb[idx + 1] = g;
                rgb[idx + 2] = b;
            }
        }
    }

    private static int clampCoord(int value, int size) {
        if (value < 0) {
            return 0;
        }
        if (value >= size) {
            return size - 1;
        }
        return value;
    }

    private static int clamp255(int value) {
        if (value < 0) {
            return 0;
        }
        if (value > 255) {
            return 255;
        }
        return value;
    }

    private record InitialState(
            float[] x, float[] y, float[] vx, float[] vy, float[] mass, float[] pairMask) {}

    private record NBodyState(float[] x, float[] y, float[] vx, float[] vy) {}
}

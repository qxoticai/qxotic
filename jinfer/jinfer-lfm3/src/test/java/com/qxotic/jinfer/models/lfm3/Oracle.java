package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads the llama.cpp dumps of {@code jinfer/scripts/lfm3-oracle.cpp} under {@code
 * test-fixtures/lfm3/oracle} (gitignored; a test without its dump skips) and scores jinfer's
 * activations against them.
 */
final class Oracle {

    static final Path ROOT =
            Path.of(System.getProperty("lfm3.oracle", "../../test-fixtures/lfm3/oracle"));

    private Oracle() {}

    /** A dump directory, or the test skips with the command that makes it. */
    static Path dir(String name) {
        Path dir = ROOT.resolve(name);
        assumeTrue(
                Files.isDirectory(dir),
                "no llama.cpp dump at " + dir + " - see jinfer/scripts/lfm3-oracle.cpp");
        return dir;
    }

    /** A graph tensor: {@code ne} in ggml order (ne[0] fastest) and its F32 values. */
    record Tensor(long[] ne, float[] data) {
        int rows() {
            return Math.toIntExact(data.length / ne[0]);
        }

        int width() {
            return Math.toIntExact(ne[0]);
        }
    }

    static Tensor tensor(Path dir, String name) throws IOException {
        ByteBuffer bytes =
                ByteBuffer.wrap(Files.readAllBytes(dir.resolve(name + ".t")))
                        .order(ByteOrder.LITTLE_ENDIAN);
        long[] ne = new long[4];
        for (int i = 0; i < 4; i++) ne[i] = bytes.getLong();
        float[] data = new float[bytes.remaining() / Float.BYTES];
        bytes.asFloatBuffer().get(data);
        return new Tensor(ne, data);
    }

    static float[] floats(Path file) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        float[] data = new float[bytes.remaining() / Float.BYTES];
        bytes.asFloatBuffer().get(data);
        return data;
    }

    static int[] ints(Path file) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        int[] data = new int[bytes.remaining() / Integer.BYTES];
        bytes.asIntBuffer().get(data);
        return data;
    }

    static float[] rows(MemoryView<MemorySegment> view, int rows, int width) {
        float[] out = new float[rows * width];
        Views.copyToArray(view, 0, out, 0, out.length, "rows");
        return out;
    }

    /**
     * How far {@code actual} is from {@code expected}, both {@code [rows][width]}: the largest
     * error relative to the reference's largest magnitude, and the worst row cosine.
     */
    record Score(double maxRel, double minCos) {
        @Override
        public String toString() {
            return String.format("maxRel %.2e  minCos %.9f", maxRel, minCos);
        }
    }

    static Score score(float[] expected, float[] actual, int width) {
        if (expected.length != actual.length)
            throw new AssertionError(expected.length + " != " + actual.length + " values");
        double maxAbs = 0, maxErr = 0, minCos = 1;
        for (float v : expected) maxAbs = Math.max(maxAbs, Math.abs(v));
        for (int r = 0; r < expected.length / width; r++) {
            double dot = 0, ee = 0, aa = 0;
            for (int i = r * width; i < (r + 1) * width; i++) {
                maxErr = Math.max(maxErr, Math.abs(expected[i] - actual[i]));
                dot += (double) expected[i] * actual[i];
                ee += (double) expected[i] * expected[i];
                aa += (double) actual[i] * actual[i];
            }
            if (ee > 0 && aa > 0) minCos = Math.min(minCos, dot / Math.sqrt(ee * aa));
        }
        return new Score(maxAbs == 0 ? maxErr : maxErr / maxAbs, minCos);
    }

    /** {@code KL(p || q)} of the softmaxes of two logit rows. */
    static double kl(float[] p, int pAt, float[] q, int qAt, int n) {
        double pMax = Double.NEGATIVE_INFINITY, qMax = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            pMax = Math.max(pMax, p[pAt + i]);
            qMax = Math.max(qMax, q[qAt + i]);
        }
        double pSum = 0, qSum = 0;
        for (int i = 0; i < n; i++) {
            pSum += Math.exp(p[pAt + i] - pMax);
            qSum += Math.exp(q[qAt + i] - qMax);
        }
        double kl = 0;
        for (int i = 0; i < n; i++) {
            double lp = p[pAt + i] - pMax - Math.log(pSum), lq = q[qAt + i] - qMax - Math.log(qSum);
            kl += Math.exp(lp) * (lp - lq);
        }
        return kl;
    }

    static int argmax(float[] values, int at, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) if (values[at + i] > values[at + best]) best = i;
        return best;
    }
}

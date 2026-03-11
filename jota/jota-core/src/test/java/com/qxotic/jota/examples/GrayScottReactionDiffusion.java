package com.qxotic.jota.examples;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.PpmWriter;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import org.junit.jupiter.api.Test;

/**
 * Gray-Scott Reaction-Diffusion simulation using Jota's tensor API.
 *
 * <p>This demo simulates the Gray-Scott model, a system of two coupled reaction-diffusion equations
 * that produce beautiful organic patterns resembling coral, fingerprints, and bacterial colonies.
 *
 * <p>The model tracks two chemical concentrations:
 *
 * <ul>
 *   <li>U - "feed" chemical (supplied from outside)
 *   <li>V - "kill" chemical (produced by reaction)
 * </ul>
 *
 * <p>Different parameter combinations produce different patterns:
 *
 * <ul>
 *   <li>Coral: F=0.0545, K=0.062 (growing spots)
 *   <li>Fingerprint: F=0.037, K=0.06 (maze-like)
 *   <li>Spots: F=0.025, K=0.06 (stable dots)
 *   <li>Worms: F=0.078, K=0.061 (traveling waves)
 * </ul>
 */
public class GrayScottReactionDiffusion {

    // Grid dimensions
    private static final int WIDTH = 256;
    private static final int HEIGHT = 256;

    // Simulation parameters
    private static final int MAX_ITER = 500000;
    private static final int SAVE_EVERY = 1000;
    private static final boolean SAVE_INTERMEDIATE_FRAMES =
            Boolean.getBoolean("jota.examples.saveIntermediateFrames");
    private static final int CONSOLE_EVERY = 20;
    private static final int CONSOLE_DELAY_MS = 20;

    // Pattern presets
    public enum Pattern {
        CORAL(0.0545f, 0.062f, "coral"),
        FINGERPRINT(0.037f, 0.06f, "fingerprint"),
        SPOTS(0.025f, 0.06f, "spots"),
        WORMS(0.078f, 0.061f, "worms");

        final float feedRate;
        final float killRate;
        final String name;

        Pattern(float feedRate, float killRate, String name) {
            this.feedRate = feedRate;
            this.killRate = killRate;
            this.name = name;
        }
    }

    public enum RenderMode {
        PPM,
        CONSOLE,
        BOTH,
        SWING
    }

    // Diffusion coefficients
    private static final float DU = 0.16f; // Diffusion rate for U
    private static final float DV = 0.08f; // Diffusion rate for V

    public static void main(String[] args) throws IOException {
        Pattern pattern = Pattern.CORAL;
        RenderMode renderMode = RenderMode.CONSOLE;
        for (String arg : args) {
            if (arg == null || arg.isBlank()) {
                continue;
            }
            if (arg.startsWith("--render=")) {
                String value = arg.substring("--render=".length()).trim();
                renderMode = parseRenderMode(value);
                if (renderMode == null) {
                    System.out.println("Unknown render mode: " + value);
                    System.out.println("Available: console, ppm, both");
                    return;
                }
                continue;
            }
            if (arg.equalsIgnoreCase("--console")) {
                renderMode = RenderMode.CONSOLE;
                continue;
            }
            if (arg.equalsIgnoreCase("--ppm")) {
                renderMode = RenderMode.PPM;
                continue;
            }
            if (arg.equalsIgnoreCase("--both")) {
                renderMode = RenderMode.BOTH;
                continue;
            }
            if (arg.equalsIgnoreCase("--swing") || arg.equalsIgnoreCase("--awt")) {
                renderMode = RenderMode.SWING;
                continue;
            }
            try {
                pattern = Pattern.valueOf(arg.toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                System.out.println("Unknown pattern: " + arg);
                System.out.println("Available patterns: coral, fingerprint, spots, worms");
                System.out.println("Render modes: --render=console|ppm|both|swing");
                return;
            }
        }

        System.out.println("Running Gray-Scott Reaction-Diffusion");
        System.out.println("Pattern: " + pattern.name);
        System.out.println("Feed rate (F): " + pattern.feedRate);
        System.out.println("Kill rate (K): " + pattern.killRate);
        System.out.println("Render: " + renderMode.name().toLowerCase(Locale.ROOT));
        System.out.println("Grid: " + WIDTH + "x" + HEIGHT);
        System.out.println("Iterations: " + MAX_ITER);

        long start = System.currentTimeMillis();

        runSimulation(pattern, renderMode);

        long elapsed = System.currentTimeMillis() - start;
        System.out.println("\nCompleted in " + elapsed + "ms");
    }

    @Test
    void testGrayScottCoral() throws IOException {
        System.out.println("\n=== Testing Gray-Scott Coral Pattern ===");
        runSimulation(Pattern.CORAL, RenderMode.PPM, 500);
    }

    @Test
    void testGrayScottFingerprint() throws IOException {
        System.out.println("\n=== Testing Gray-Scott Fingerprint Pattern ===");
        runSimulation(Pattern.FINGERPRINT, RenderMode.PPM, 500);
    }

    @Test
    void testGrayScottSpots() throws IOException {
        System.out.println("\n=== Testing Gray-Scott Spots Pattern ===");
        runSimulation(Pattern.SPOTS, RenderMode.PPM, 500);
    }

    private static void runSimulation(Pattern pattern, RenderMode renderMode) throws IOException {
        runSimulation(pattern, renderMode, MAX_ITER);
    }

    private static void runSimulation(Pattern pattern, RenderMode renderMode, int iterations)
            throws IOException {
        Shape shape = Shape.of(HEIGHT, WIDTH);

        // Initialize U and V with seed in center
        float[] u = new float[HEIGHT * WIDTH];
        float[] v = new float[HEIGHT * WIDTH];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = y * WIDTH + x;

                // Square seed in center
                int seedSize = WIDTH / 10;
                int cx = WIDTH / 2;
                int cy = HEIGHT / 2;

                if (x >= cx - seedSize
                        && x <= cx + seedSize
                        && y >= cy - seedSize
                        && y <= cy + seedSize) {
                    u[idx] = 0.5f;
                    v[idx] = 0.25f;
                } else {
                    u[idx] = 1.0f;
                    v[idx] = 0.0f;
                }
            }
        }

        // Pre-create constant tensors
        Tensor zeros = Tensor.zeros(DataType.FP32, shape);
        Tensor ones = Tensor.ones(DataType.FP32, shape);
        Tensor fTensor = Tensor.full(pattern.feedRate, DataType.FP32, shape);
        Tensor fkTensor = Tensor.full(pattern.feedRate + pattern.killRate, DataType.FP32, shape);
        Tensor duTensor = Tensor.full(DU, DataType.FP32, shape);
        Tensor dvTensor = Tensor.full(DV, DataType.FP32, shape);

        if (renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH) {
            System.out.print("\033[2J\033[H\033[?25l");
        }
        SwingRenderer swingRenderer = null;
        if (renderMode == RenderMode.SWING) {
            swingRenderer = new SwingRenderer(WIDTH, HEIGHT, "Gray-Scott");
        }

        // Run simulation
        for (int iter = 0; iter < iterations; iter++) {
            // Compute laplacians using Java (simpler and faster for 5-point stencil)
            float[] lapU = computeLaplacianJava(u);
            float[] lapV = computeLaplacianJava(v);

            // Create tensors for this iteration
            Tensor uTensor = Tensor.of(u.clone(), shape);
            Tensor vTensor = Tensor.of(v.clone(), shape);
            Tensor lapUTensor = Tensor.of(lapU, shape);
            Tensor lapVTensor = Tensor.of(lapV, shape);

            // Gray-Scott reaction terms: uvv = u * v * v
            Tensor uvv = uTensor.multiply(vTensor).multiply(vTensor);

            // dU = Du * laplacianU - uvv + F * (1 - u)
            Tensor dU =
                    duTensor.multiply(lapUTensor)
                            .subtract(uvv)
                            .add(fTensor.multiply(uTensor.negate().add(1.0f)));

            // dV = Dv * laplacianV + uvv - (F + K) * v
            Tensor dV = dvTensor.multiply(lapVTensor).add(uvv).subtract(fkTensor.multiply(vTensor));

            // Update: u = u + dU, v = v + dV
            Tensor newU = uTensor.add(dU);
            Tensor newV = vTensor.add(dV);

            // Clamp to [0, 1]
            newU = newU.max(zeros).min(ones);
            newV = newV.max(zeros).min(ones);

            // Materialize back to arrays
            u = tensorToFloatArray(newU);
            v = tensorToFloatArray(newV);

            // Save intermediate frames
            if ((renderMode == RenderMode.PPM || renderMode == RenderMode.BOTH)
                    && SAVE_INTERMEDIATE_FRAMES
                    && iter > 0
                    && iter % SAVE_EVERY == 0) {
                String filename = String.format("target/grayscott-%s-%05d.ppm", pattern.name, iter);
                saveVisualization(v, filename);
                System.out.println("Iteration " + iter + ", Saved: " + filename);
            }

            if ((renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH)
                    && iter % CONSOLE_EVERY == 0) {
                renderConsole(v, iter, pattern);
            }
            if (renderMode == RenderMode.SWING && iter % CONSOLE_EVERY == 0) {
                swingRenderer.renderField(
                        v, iter, pattern.name, pattern.feedRate, pattern.killRate);
            }
        }

        if (renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH) {
            renderConsole(v, iterations, pattern);
            System.out.print("\033[?25h");
        }

        if (renderMode == RenderMode.PPM || renderMode == RenderMode.BOTH) {
            String finalFilename = String.format("target/grayscott-%s-final.ppm", pattern.name);
            saveVisualization(v, finalFilename);
            System.out.println("Saved: " + finalFilename);
        }
        if (renderMode == RenderMode.SWING) {
            swingRenderer.renderField(
                    v, iterations, pattern.name, pattern.feedRate, pattern.killRate);
            swingRenderer.waitUntilClosed();
        }
    }

    private static RenderMode parseRenderMode(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "console" -> RenderMode.CONSOLE;
            case "ppm" -> RenderMode.PPM;
            case "both" -> RenderMode.BOTH;
            case "swing", "awt" -> RenderMode.SWING;
            default -> null;
        };
    }

    private static void renderConsole(float[] v, int iter, Pattern pattern) {
        boolean color = terminalSupportsColor();
        int cols = terminalEnvInt("COLUMNS", 120);
        int rows = terminalEnvInt("LINES", 40);
        int drawWidth =
                color
                        ? Math.max(16, Math.min(WIDTH, (cols - 2) / 2))
                        : Math.max(16, Math.min(WIDTH, cols - 2));
        int drawHeight = Math.max(8, Math.min(HEIGHT, rows - 5));

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float value : v) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
        float span = Math.max(1.0e-6f, max - min);
        char[] ramp = " .:-=+*#%@".toCharArray();

        StringBuilder frame = new StringBuilder(drawWidth * drawHeight + 128);
        frame.append("\033[H");
        frame.append("Gray-Scott [").append(pattern.name).append("]");
        frame.append(" iter=").append(iter).append('/').append(MAX_ITER).append('\n');
        frame.append("v-range=")
                .append(String.format(Locale.ROOT, "%.4f..%.4f", min, max))
                .append("  F=")
                .append(pattern.feedRate)
                .append("  K=")
                .append(pattern.killRate)
                .append("  view=")
                .append(color ? "ansi-color" : "ascii")
                .append('\n');

        for (int y = 0; y < drawHeight; y++) {
            int srcY = y * HEIGHT / drawHeight;
            for (int x = 0; x < drawWidth; x++) {
                int srcX = x * WIDTH / drawWidth;
                float value = v[srcY * WIDTH + srcX];
                float norm = (value - min) / span;
                if (norm < 0f) {
                    norm = 0f;
                }
                if (norm > 1f) {
                    norm = 1f;
                }

                if (color) {
                    int r = (int) (Math.pow(norm, 0.45) * 255.0);
                    int g = (int) (Math.pow(norm, 0.9) * 220.0 + 15.0);
                    int b = (int) (Math.pow(1.0 - norm, 1.7) * 255.0);
                    appendAnsiBg(frame, clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255));
                    frame.append("  ");
                } else {
                    int idx = (int) (norm * (ramp.length - 1));
                    if (idx < 0) {
                        idx = 0;
                    }
                    if (idx >= ramp.length) {
                        idx = ramp.length - 1;
                    }
                    frame.append(ramp[idx]);
                }
            }
            if (color) {
                frame.append("\033[0m");
            }
            frame.append('\n');
        }

        System.out.print(frame);
        sleepQuietly(CONSOLE_DELAY_MS);
    }

    private static int terminalEnvInt(String name, int fallback) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static boolean terminalSupportsColor() {
        if (System.getenv("NO_COLOR") != null) {
            return false;
        }
        String term = System.getenv("TERM");
        if (term == null || term.isBlank() || "dumb".equalsIgnoreCase(term)) {
            return false;
        }
        String colorterm = System.getenv("COLORTERM");
        return colorterm != null
                || term.contains("color")
                || term.contains("256")
                || term.contains("xterm");
    }

    private static void appendAnsiBg(StringBuilder builder, int r, int g, int b) {
        builder.append("\033[48;2;")
                .append(r)
                .append(';')
                .append(g)
                .append(';')
                .append(b)
                .append('m');
    }

    private static void sleepQuietly(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static final class SwingRenderer {
        private final int width;
        private final int height;
        private final AtomicBoolean open = new AtomicBoolean(true);
        private final BufferedImage image;
        private final JFrame frame;
        private final JPanel panel;

        private SwingRenderer(int width, int height, String title) {
            this.width = width;
            this.height = height;
            this.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            CountDownLatch latch = new CountDownLatch(1);
            JFrame[] frameRef = new JFrame[1];
            JPanel[] panelRef = new JPanel[1];
            SwingUtilities.invokeLater(
                    () -> {
                        JFrame f = new JFrame(title);
                        JPanel p =
                                new JPanel() {
                                    @Override
                                    protected void paintComponent(Graphics g) {
                                        super.paintComponent(g);
                                        Graphics2D g2 = (Graphics2D) g;
                                        g2.setRenderingHint(
                                                RenderingHints.KEY_INTERPOLATION,
                                                RenderingHints
                                                        .VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
                                        g2.drawImage(image, 0, 0, getWidth(), getHeight(), null);
                                    }
                                };
                        p.setPreferredSize(new Dimension(width * 3, height * 3));
                        f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
                        f.addWindowListener(
                                new WindowAdapter() {
                                    @Override
                                    public void windowClosed(WindowEvent e) {
                                        open.set(false);
                                    }
                                });
                        f.setContentPane(p);
                        f.pack();
                        f.setLocationByPlatform(true);
                        f.setVisible(true);
                        frameRef[0] = f;
                        panelRef[0] = p;
                        latch.countDown();
                    });
            awaitLatch(latch);
            this.frame = frameRef[0];
            this.panel = panelRef[0];
        }

        private void renderField(float[] v, int iter, String patternName, float feed, float kill) {
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (float value : v) {
                if (value < min) {
                    min = value;
                }
                if (value > max) {
                    max = value;
                }
            }
            float span = Math.max(1.0e-6f, max - min);
            int[] rgb = new int[width * height];
            for (int i = 0; i < v.length; i++) {
                float norm = (v[i] - min) / span;
                if (norm < 0f) {
                    norm = 0f;
                }
                if (norm > 1f) {
                    norm = 1f;
                }
                int r = clamp((int) (Math.pow(norm, 0.45) * 255.0), 0, 255);
                int g = clamp((int) (Math.pow(norm, 0.9) * 220.0 + 15.0), 0, 255);
                int b = clamp((int) (Math.pow(1.0 - norm, 1.7) * 255.0), 0, 255);
                rgb[i] = (r << 16) | (g << 8) | b;
            }
            SwingUtilities.invokeLater(
                    () -> {
                        image.setRGB(0, 0, width, height, rgb, 0, width);
                        frame.setTitle(
                                "Gray-Scott - "
                                        + patternName
                                        + " - iter "
                                        + iter
                                        + "/"
                                        + MAX_ITER
                                        + " - F="
                                        + feed
                                        + " K="
                                        + kill);
                        panel.repaint();
                    });
            sleepQuietly(CONSOLE_DELAY_MS);
        }

        private void waitUntilClosed() {
            while (open.get()) {
                sleepQuietly(100);
            }
        }

        private static void awaitLatch(CountDownLatch latch) {
            try {
                latch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Computes the discrete Laplacian using a 5-point stencil with periodic boundaries. */
    private static float[] computeLaplacianJava(float[] field) {
        float[] laplacian = new float[HEIGHT * WIDTH];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = y * WIDTH + x;
                float center = field[idx];

                // 5-point stencil with wrap-around boundaries
                int up = ((y - 1 + HEIGHT) % HEIGHT) * WIDTH + x;
                int down = ((y + 1) % HEIGHT) * WIDTH + x;
                int left = y * WIDTH + ((x - 1 + WIDTH) % WIDTH);
                int right = y * WIDTH + ((x + 1) % WIDTH);

                laplacian[idx] = field[up] + field[down] + field[left] + field[right] - 4 * center;
            }
        }

        return laplacian;
    }

    @SuppressWarnings("unchecked")
    private static float[] tensorToFloatArray(Tensor tensor) {
        MemoryView<MemorySegment> view = (MemoryView<MemorySegment>) tensor.materialize();
        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();

        int size = (int) view.shape().size();
        float[] result = new float[size];

        for (int i = 0; i < size; i++) {
            long offset = Indexing.linearToOffset(view, i);
            result[i] = access.readFloat(view.memory(), offset);
        }

        return result;
    }

    private static void saveVisualization(float[] v, String filename) throws IOException {
        byte[] rgb = new byte[WIDTH * HEIGHT * 3];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = y * WIDTH + x;
                float value = v[idx];

                double t = value;
                rgb[idx * 3] = (byte) clamp((int) (Math.pow(t, 0.5) * 255), 0, 255);
                rgb[idx * 3 + 1] = (byte) clamp((int) (t * 200), 0, 255);
                rgb[idx * 3 + 2] = (byte) clamp((int) (Math.pow(1 - t, 2) * 255), 0, 255);
            }
        }
        PpmWriter.write(Path.of(filename), WIDTH, HEIGHT, rgb);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}

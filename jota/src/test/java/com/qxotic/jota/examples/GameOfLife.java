package com.qxotic.jota.examples;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.MemorySegment;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import org.junit.jupiter.api.Test;

/**
 * Conway's Game of Life using Jota's tensor API.
 *
 * <p>This demo implements the classic cellular automaton using tensor operations for parallel
 * neighbor counting. The game follows these rules on a 2D grid of cells:
 *
 * <ul>
 *   <li>Any live cell with 2-3 live neighbors survives
 *   <li>Any dead cell with exactly 3 live neighbors becomes alive (birth)
 *   <li>All other cells die or stay dead
 * </ul>
 *
 * <p>The implementation uses tensor slicing to count 8 neighbors for each cell in parallel, then
 * applies the game rules using logical tensor operations.
 *
 * <p>Includes several interesting patterns:
 *
 * <ul>
 *   <li>Random: Random initial state
 *   <li>Glider: Simple moving pattern
 *   <li>Blinker: Period-2 oscillator
 *   <li>GliderGun: Gosper's glider gun (infinite pattern generator)
 *   <li>Pulsar: Period-3 oscillator
 * </ul>
 */
public class GameOfLife {

    // Grid dimensions
    private static final int WIDTH = 1024;
    private static final int HEIGHT = 1024;

    // Simulation parameters
    private static final int MAX_ITER = 20000;
    private static final int SAVE_EVERY = 20;
    private static final int CONSOLE_EVERY = 1;
    private static final int CONSOLE_DELAY_MS = 35;

    // Pattern types
    public enum Pattern {
        RANDOM("random"),
        GLIDER("glider"),
        BLINKER("blinker"),
        PULSAR("pulsar"),
        GLIDER_GUN("glidergun");

        final String name;

        Pattern(String name) {
            this.name = name;
        }
    }

    public enum RenderMode {
        PPM,
        CONSOLE,
        BOTH,
        SWING
    }

    public static void main(String[] args) throws IOException {
        Pattern pattern = Pattern.RANDOM;
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
                System.out.println(
                        "Available patterns: random, glider, blinker, pulsar, glidergun");
                System.out.println("Render modes: --render=console|ppm|both|swing");
                return;
            }
        }

        System.out.println("Running Conway's Game of Life");
        System.out.println("Pattern: " + pattern.name);
        System.out.println("Render: " + renderMode.name().toLowerCase(Locale.ROOT));
        System.out.println("Grid: " + WIDTH + "x" + HEIGHT);
        System.out.println("Iterations: " + MAX_ITER);

        long start = System.currentTimeMillis();

        runSimulation(pattern, renderMode);

        long elapsed = System.currentTimeMillis() - start;
        System.out.println("\nCompleted in " + elapsed + "ms");
    }

    @Test
    void testGameOfLifeRandom() throws IOException {
        System.out.println("\n=== Testing Game of Life - Random ===");
        runSimulation(Pattern.RANDOM, RenderMode.PPM);
    }

    @Test
    void testGameOfLifeGlider() throws IOException {
        System.out.println("\n=== Testing Game of Life - Glider ===");
        runSimulation(Pattern.GLIDER, RenderMode.PPM);
    }

    @Test
    void testGameOfLifeGliderGun() throws IOException {
        System.out.println("\n=== Testing Game of Life - Glider Gun ===");
        runSimulation(Pattern.GLIDER_GUN, RenderMode.PPM);
    }

    private static void runSimulation(Pattern pattern, RenderMode renderMode) throws IOException {
        Shape shape = Shape.of(HEIGHT, WIDTH);

        // Initialize grid based on pattern
        float[] gridData = initializePattern(pattern);
        int[] ageData = initializeAges(gridData);

        // Run simulation
        if (renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH) {
            System.out.print("\033[2J\033[H\033[?25l");
        }
        SwingRenderer swingRenderer = null;
        if (renderMode == RenderMode.SWING) {
            swingRenderer = new SwingRenderer(WIDTH, HEIGHT, "Game of Life");
        }

        for (int iter = 0; iter < MAX_ITER; iter++) {
            // Create tensor from current grid
            Tensor grid = Tensor.of(gridData.clone(), shape);

            // Count neighbors
            float[] neighborData = countNeighborsJava(gridData);
            Tensor neighbors = Tensor.of(neighborData, shape);

            // Game of Life rules using tensor operations
            Tensor zero = Tensor.zeros(DataType.FP32, shape);
            Tensor one = Tensor.ones(DataType.FP32, shape);
            Tensor two = Tensor.full(2f, DataType.FP32, shape);
            Tensor three = Tensor.full(3f, DataType.FP32, shape);

            // A cell is alive in the next generation if:
            //   (alive AND neighbors in [2,3]) OR (dead AND neighbors == 3)
            Tensor isAlive = grid.greaterThan(zero);
            Tensor n2 = neighbors.equal(two);
            Tensor n3 = neighbors.equal(three);

            // Survive: alive AND (neighbors == 2 OR neighbors == 3)
            Tensor survive = isAlive.logicalAnd(n2.logicalOr(n3));

            // Birth: dead AND neighbors == 3
            Tensor birth = isAlive.logicalNot().logicalAnd(n3);

            // New grid = survive OR birth
            Tensor newGrid = survive.select(one, birth.select(one, zero));

            // Materialize to get next state
            gridData = tensorToFloatArray(newGrid);
            updateAges(ageData, gridData);

            // Save intermediate frames
            if ((renderMode == RenderMode.PPM || renderMode == RenderMode.BOTH)
                    && iter % SAVE_EVERY == 0) {
                String filename =
                        String.format("target/gameoflife-%s-%04d.ppm", pattern.name, iter);
                saveFrame(gridData, filename);
                System.out.println("Iteration " + iter + ", Saved: " + filename);
            }

            if ((renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH)
                    && iter % CONSOLE_EVERY == 0) {
                renderConsole(gridData, iter, pattern);
            }
            if (renderMode == RenderMode.SWING && iter % CONSOLE_EVERY == 0) {
                swingRenderer.renderColored(gridData, ageData, iter, pattern.name);
            }
        }

        if (renderMode == RenderMode.CONSOLE || renderMode == RenderMode.BOTH) {
            renderConsole(gridData, MAX_ITER, pattern);
            System.out.print("\033[?25h");
        }

        if (renderMode == RenderMode.PPM || renderMode == RenderMode.BOTH) {
            String finalFilename = String.format("target/gameoflife-%s-final.ppm", pattern.name);
            saveFrame(gridData, finalFilename);
            System.out.println("Saved: " + finalFilename);
        }
        if (renderMode == RenderMode.SWING) {
            swingRenderer.renderColored(gridData, ageData, MAX_ITER, pattern.name);
            swingRenderer.waitUntilClosed();
        }
    }

    private static int[] initializeAges(float[] gridData) {
        int[] ageData = new int[gridData.length];
        for (int i = 0; i < gridData.length; i++) {
            ageData[i] = gridData[i] > 0.5f ? 1 : 0;
        }
        return ageData;
    }

    private static void updateAges(int[] ageData, float[] gridData) {
        for (int i = 0; i < gridData.length; i++) {
            if (gridData[i] > 0.5f) {
                ageData[i] = Math.min(ageData[i] + 1, 1024);
            } else {
                ageData[i] = 0;
            }
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

    private static void renderConsole(float[] gridData, int iter, Pattern pattern) {
        boolean color = terminalSupportsColor();
        int cols = terminalEnvInt("COLUMNS", 120);
        int rows = terminalEnvInt("LINES", 40);
        int drawWidth =
                color
                        ? Math.max(16, Math.min(WIDTH, (cols - 2) / 2))
                        : Math.max(16, Math.min(WIDTH, cols - 2));
        int drawHeight = Math.max(8, Math.min(HEIGHT, rows - 5));

        StringBuilder frame = new StringBuilder(drawWidth * drawHeight + 128);
        frame.append("\033[H");
        frame.append("Game of Life [").append(pattern.name).append("]");
        frame.append(" iter=").append(iter).append('/').append(MAX_ITER).append('\n');

        int alive = 0;
        for (float v : gridData) {
            if (v > 0.5f) {
                alive++;
            }
        }
        frame.append("alive=")
                .append(alive)
                .append("  density=")
                .append(String.format(Locale.ROOT, "%.2f%%", 100.0 * alive / (WIDTH * HEIGHT)))
                .append("  view=")
                .append(color ? "ansi-color" : "ascii")
                .append('\n');

        for (int y = 0; y < drawHeight; y++) {
            int srcY = y * HEIGHT / drawHeight;
            for (int x = 0; x < drawWidth; x++) {
                int srcX = x * WIDTH / drawWidth;
                float v = gridData[srcY * WIDTH + srcX];
                if (color) {
                    if (v > 0.5f) {
                        appendAnsiBg(frame, 62, 190, 121);
                    } else {
                        appendAnsiBg(frame, 16, 22, 26);
                    }
                    frame.append("  ");
                } else {
                    frame.append(v > 0.5f ? '#' : '.');
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
                        p.setPreferredSize(new Dimension(width * 4, height * 4));
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

        private void renderColored(float[] data, int[] ages, int iter, String patternName) {
            int[] rgb = new int[width * height];
            for (int i = 0; i < data.length; i++) {
                if (data[i] > 0.5f) {
                    int age = ages[i];
                    float maturity = Math.min(1.0f, age / 40.0f);
                    float hue = (age % 180) / 180.0f;
                    float saturation = 0.75f + 0.2f * maturity;
                    float brightness = 0.45f + 0.5f * maturity;
                    rgb[i] = Color.HSBtoRGB(hue, saturation, brightness);
                } else {
                    rgb[i] = 0xFF0C1218;
                }
            }
            SwingUtilities.invokeLater(
                    () -> {
                        image.setRGB(0, 0, width, height, rgb, 0, width);
                        frame.setTitle(
                                "Game of Life (Color) - "
                                        + patternName
                                        + " - iter "
                                        + iter
                                        + "/"
                                        + MAX_ITER);
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

    /** Counts neighbors using Java array operations (faster and simpler for this use case). */
    private static float[] countNeighborsJava(float[] grid) {
        float[] neighbors = new float[HEIGHT * WIDTH];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int count = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;

                        int ny = y + dy;
                        int nx = x + dx;

                        // Wrap around boundaries (toroidal)
                        if (ny < 0) ny = HEIGHT - 1;
                        if (ny >= HEIGHT) ny = 0;
                        if (nx < 0) nx = WIDTH - 1;
                        if (nx >= WIDTH) nx = 0;

                        if (grid[ny * WIDTH + nx] > 0.5f) {
                            count++;
                        }
                    }
                }
                neighbors[y * WIDTH + x] = count;
            }
        }

        return neighbors;
    }

    private static float[] initializePattern(Pattern pattern) {
        float[] data = new float[HEIGHT * WIDTH];

        switch (pattern) {
            case RANDOM:
                // Random initial state (30% alive)
                java.util.Random random =
                        new java.util.Random(42); // Fixed seed for reproducibility
                for (int i = 0; i < data.length; i++) {
                    data[i] = random.nextFloat() < 0.3f ? 1.0f : 0.0f;
                }
                break;

            case GLIDER:
                // Simple glider pattern
                int cx = WIDTH / 2;
                int cy = HEIGHT / 2;
                setCell(data, cx, cy - 1, 1); // top
                setCell(data, cx + 1, cy, 1); // right
                setCell(data, cx - 1, cy + 1, 1); // bottom-left
                setCell(data, cx, cy + 1, 1); // bottom
                setCell(data, cx + 1, cy + 1, 1); // bottom-right
                break;

            case BLINKER:
                // Period-2 oscillator
                cx = WIDTH / 2;
                cy = HEIGHT / 2;
                setCell(data, cx - 1, cy, 1);
                setCell(data, cx, cy, 1);
                setCell(data, cx + 1, cy, 1);
                break;

            case PULSAR:
                // Period-3 oscillator
                cx = WIDTH / 2;
                cy = HEIGHT / 2;
                int[][] pulsarPattern = {
                    {-4, -2}, {-4, -1}, {-4, 0}, {-4, 1}, {-4, 2}, {-2, -4}, {-1, -4}, {0, -4},
                    {1, -4}, {2, -4}, {2, -2}, {2, -1}, {2, 0}, {2, 1}, {2, 2}, {-2, 4}, {-1, 4},
                    {0, 4}, {1, 4}, {2, 4}, {4, -2}, {4, -1}, {4, 0}, {4, 1}, {4, 2}, {-2, 2},
                    {-1, 2}, {0, 2}, {1, 2}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}
                };
                for (int[] p : pulsarPattern) {
                    setCell(data, cx + p[0], cy + p[1], 1);
                }
                break;

            case GLIDER_GUN:
                // Gosper's glider gun
                int offsetX = 10;
                int offsetY = HEIGHT / 2;
                int[][] gunPattern = {
                    {0, 4},
                    {0, 5},
                    {1, 4},
                    {1, 5}, // Block
                    {10, 4},
                    {10, 5},
                    {10, 6},
                    {11, 3},
                    {11, 7},
                    {12, 2},
                    {12, 8},
                    {13, 2},
                    {13, 8},
                    {14, 5},
                    {15, 3},
                    {15, 7},
                    {16, 4},
                    {16, 5},
                    {16, 6},
                    {17, 5}, // Left shuttle
                    {20, 2},
                    {20, 3},
                    {20, 4},
                    {21, 2},
                    {21, 3},
                    {21, 4},
                    {22, 1},
                    {22, 5},
                    {24, 0},
                    {24, 1},
                    {24, 5},
                    {24, 6}, // Right shuttle
                    {34, 2},
                    {34, 3},
                    {35, 2},
                    {35, 3} // Block
                };
                for (int[] p : gunPattern) {
                    setCell(data, offsetX + p[0], offsetY + p[1], 1);
                }
                break;
        }

        return data;
    }

    private static void setCell(float[] data, int x, int y, float value) {
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            data[y * WIDTH + x] = value;
        }
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

    private static void saveFrame(float[] gridData, String filename) throws IOException {
        try (PrintStream out =
                new PrintStream(new BufferedOutputStream(new FileOutputStream(filename)))) {
            out.println("P3");
            out.println(WIDTH + " " + HEIGHT);
            out.println("255");

            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int idx = y * WIDTH + x;
                    float value = gridData[idx];

                    // Alive = white, Dead = black
                    int shade = value > 0.5f ? 255 : 0;

                    out.print(shade + " " + shade + " " + shade);
                    if (x < WIDTH - 1) {
                        out.print(" ");
                    }
                }
                out.println();
            }
        }
    }
}

package com.qxotic.jota.examples.demos;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.tensor.Tensor;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

public final class NBodySwingDemo {

    private static final int DEFAULT_WIDTH = 1100;
    private static final int DEFAULT_HEIGHT = 800;
    private static final int DEFAULT_BODIES = 256;
    private static final float DEFAULT_DT = 0.0035f;
    private static final float DEFAULT_G = 0.06f;
    private static final float DEFAULT_MOUSE_G = 18.0f;
    private static final float DEFAULT_MOUSE_SPRING = 42.0f;
    private static final float DEFAULT_MOUSE_STEER = 60.0f;
    private static final float DEFAULT_SOFTENING = 0.0008f;
    private static final float DEFAULT_DAMPING = 0.9992f;
    private static final float DEFAULT_WORLD_RADIUS = 1.0f;

    private NBodySwingDemo() {}

    public static void main(String[] args) {
        int bodies = intArg(args, "--bodies", DEFAULT_BODIES);
        int width = intArg(args, "--width", DEFAULT_WIDTH);
        int height = intArg(args, "--height", DEFAULT_HEIGHT);
        long seed = longArg(args, "--seed", 42L);
        Device device = parseDevice(stringArg(args, "--device", "panama"));
        boolean listDevices = boolFlag(args, "--list-devices");

        Environment global = Environment.global();
        if (listDevices) {
            printRuntimeDiagnostics(global);
            return;
        }
        if (!global.runtimes().hasRuntime(device)) {
            throw new IllegalStateException(unavailableDeviceMessage(global, device));
        }
        try {
            Environment.configureGlobal(new Environment(device, DataType.FP32, global.runtimes()));
        } catch (IllegalStateException ignored) {
        }

        SwingUtilities.invokeLater(() -> createAndShow(width, height, bodies, seed, device));
    }

    private static void createAndShow(int width, int height, int bodies, long seed, Device device) {
        NBodySimulation simulation =
                new NBodySimulation(
                        bodies,
                        DEFAULT_DT,
                        DEFAULT_G,
                        DEFAULT_MOUSE_G,
                        DEFAULT_MOUSE_SPRING,
                        DEFAULT_MOUSE_STEER,
                        DEFAULT_SOFTENING,
                        DEFAULT_DAMPING,
                        DEFAULT_WORLD_RADIUS,
                        seed);

        NBodyPanel panel = new NBodyPanel(simulation, width, height);
        JFrame frame = new JFrame("Jota Tensor N-Body Demo [" + device.runtimeId() + "]");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(panel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        Timer timer =
                new Timer(
                        16,
                        e -> {
                            simulation.step(
                                    panel.mouseWorldX(), panel.mouseWorldY(), panel.mouseInside());
                            panel.repaint();
                        });
        timer.start();
    }

    private static int intArg(String[] args, String key, int fallback) {
        for (int i = 0; i < args.length; i++) {
            if (key.equals(args[i]) && i + 1 < args.length) {
                return Integer.parseInt(args[i + 1]);
            }
            if (args[i].startsWith(key + "=")) {
                return Integer.parseInt(args[i].substring(key.length() + 1));
            }
        }
        return fallback;
    }

    private static long longArg(String[] args, String key, long fallback) {
        for (int i = 0; i < args.length; i++) {
            if (key.equals(args[i]) && i + 1 < args.length) {
                return Long.parseLong(args[i + 1]);
            }
            if (args[i].startsWith(key + "=")) {
                return Long.parseLong(args[i].substring(key.length() + 1));
            }
        }
        return fallback;
    }

    private static String stringArg(String[] args, String key, String fallback) {
        for (int i = 0; i < args.length; i++) {
            if (key.equals(args[i]) && i + 1 < args.length) {
                return args[i + 1];
            }
            if (args[i].startsWith(key + "=")) {
                return args[i].substring(key.length() + 1);
            }
        }
        return fallback;
    }

    private static boolean boolFlag(String[] args, String key) {
        for (String arg : args) {
            if (key.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static Device parseDevice(String value) {
        String normalized = value == null ? "panama" : value.trim().toLowerCase();
        return switch (normalized) {
            case "panama" -> DeviceType.PANAMA.deviceIndex(0);
            case "c" -> DeviceType.C.deviceIndex(0);
            case "hip" -> DeviceType.HIP.deviceIndex(0);
            case "opencl" -> DeviceType.OPENCL.deviceIndex(0);
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported --device value: " + value + " (use panama|c|hip|opencl)");
        };
    }

    private static void printRuntimeDiagnostics(Environment environment) {
        System.out.println("Runtime availability:");
        for (RuntimeDiagnostic diagnostic : environment.runtimeDiagnostics()) {
            String status = diagnostic.probe().status().name().toLowerCase();
            String line =
                    "- "
                            + diagnostic.deviceType().id()
                            + " ["
                            + status
                            + "] "
                            + diagnostic.probe().message();
            System.out.println(line);
            if (diagnostic.probe().hint() != null) {
                System.out.println("  hint: " + diagnostic.probe().hint());
            }
        }
    }

    private static String unavailableDeviceMessage(Environment environment, Device device) {
        StringBuilder message =
                new StringBuilder("Requested device runtime is not available: " + device + "\n");
        for (RuntimeDiagnostic diagnostic : environment.runtimeDiagnostics()) {
            if (!diagnostic.deviceType().equals(device.type())) {
                continue;
            }
            message.append("- ")
                    .append(diagnostic.probe().status().name().toLowerCase())
                    .append(": ")
                    .append(diagnostic.probe().message())
                    .append("\n");
            if (diagnostic.probe().hint() != null) {
                message.append("  hint: ").append(diagnostic.probe().hint()).append("\n");
            }
        }
        message.append("Try: --device panama or run with --list-devices for diagnostics.");
        return message.toString();
    }

    private static final class NBodyPanel extends JPanel {
        private final NBodySimulation simulation;
        private final int width;
        private final int height;
        private long lastFpsNanos;
        private int frames;
        private int fps;
        private float mouseWorldX;
        private float mouseWorldY;
        private boolean mouseInside;

        private NBodyPanel(NBodySimulation simulation, int width, int height) {
            this.simulation = simulation;
            this.width = width;
            this.height = height;
            this.lastFpsNanos = System.nanoTime();
            setBackground(new Color(7, 10, 20));
            setPreferredSize(new Dimension(width, height));
            addMouseMotionListener(
                    new MouseMotionAdapter() {
                        @Override
                        public void mouseMoved(MouseEvent e) {
                            updateMousePosition(e.getX(), e.getY());
                        }

                        @Override
                        public void mouseDragged(MouseEvent e) {
                            updateMousePosition(e.getX(), e.getY());
                        }
                    });
            addMouseListener(
                    new MouseAdapter() {
                        @Override
                        public void mouseEntered(MouseEvent e) {
                            mouseInside = true;
                            updateMousePosition(e.getX(), e.getY());
                        }

                        @Override
                        public void mouseExited(MouseEvent e) {
                            mouseInside = false;
                        }
                    });
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            float[] x = simulation.xHost;
            float[] y = simulation.yHost;
            float[] m = simulation.massHost;
            float worldRadius = simulation.worldRadius;

            int count = simulation.count;
            int cx = width / 2;
            int cy = height / 2;
            float scale = 0.48f * Math.min(width, height) / worldRadius;

            for (int i = 0; i < count; i++) {
                int px = cx + Math.round(x[i] * scale);
                int py = cy + Math.round(y[i] * scale);
                int size = Math.max(2, Math.min(8, 2 + Math.round(m[i] * 2.0f)));
                float c = Math.min(1.0f, m[i] / 2.5f);
                int red = 150 + Math.round(90 * c);
                int green = 170 + Math.round(60 * (1.0f - c));
                int blue = 230;
                g.setColor(new Color(red, green, blue));
                g.fillOval(px - size / 2, py - size / 2, size, size);
            }

            updateFps();
            g.setColor(new Color(230, 235, 255));
            g.drawString("Bodies: " + count + " | FPS: " + fps, 12, 20);
            g.drawString("Move mouse to steer gravity field", 12, 38);

            if (mouseInside) {
                int mx = cx + Math.round(mouseWorldX * scale);
                int my = cy + Math.round(mouseWorldY * scale);
                g.setColor(new Color(255, 215, 120, 190));
                g.drawOval(mx - 12, my - 12, 24, 24);
                g.drawOval(mx - 22, my - 22, 44, 44);
            }
        }

        private void updateMousePosition(int px, int py) {
            int cx = width / 2;
            int cy = height / 2;
            float scale = 0.48f * Math.min(width, height) / simulation.worldRadius;
            mouseWorldX = (px - cx) / scale;
            mouseWorldY = (py - cy) / scale;
        }

        private float mouseWorldX() {
            return mouseWorldX;
        }

        private float mouseWorldY() {
            return mouseWorldY;
        }

        private boolean mouseInside() {
            return mouseInside;
        }

        private void updateFps() {
            frames++;
            long now = System.nanoTime();
            long elapsed = now - lastFpsNanos;
            if (elapsed >= 1_000_000_000L) {
                fps = Math.round(frames * 1_000_000_000f / elapsed);
                frames = 0;
                lastFpsNanos = now;
            }
        }
    }

    private static final class NBodySimulation {
        private final int count;
        private final float dt;
        private final float g;
        private final float mouseG;
        private final float mouseSpring;
        private final float mouseSteer;
        private final float softening;
        private final float damping;
        private final float worldRadius;

        private final Shape vecShape;
        private final Shape pairShape;
        private final Tensor pairMask;
        private final Tensor onesCol;
        private final Tensor onesRow;
        private final boolean stabilizeLazyState;

        private Tensor x;
        private Tensor y;
        private Tensor vx;
        private Tensor vy;
        private final Tensor mass;

        private float[] xHost;
        private float[] yHost;
        private final float[] massHost;

        // Pre-allocated arrays for mouse interaction (avoids reallocation)
        private final float[] mouseXArr;
        private final float[] mouseYArr;

        private NBodySimulation(
                int count,
                float dt,
                float g,
                float mouseG,
                float mouseSpring,
                float mouseSteer,
                float softening,
                float damping,
                float worldRadius,
                long seed) {
            this.count = count;
            this.dt = dt;
            this.g = g;
            this.mouseG = mouseG;
            this.mouseSpring = mouseSpring;
            this.mouseSteer = mouseSteer;
            this.softening = softening;
            this.damping = damping;
            this.worldRadius = worldRadius;

            this.vecShape = Shape.of(count, 1);
            this.pairShape = Shape.of(count, count);
            this.stabilizeLazyState =
                    Environment.current().defaultDevice().belongsTo(DeviceType.HIP);

            float[] xArr = new float[count];
            float[] yArr = new float[count];
            float[] vxArr = new float[count];
            float[] vyArr = new float[count];
            float[] mArr = new float[count];
            float[] maskArr = new float[count * count];

            Random random = new Random(seed);
            for (int i = 0; i < count; i++) {
                float r = (float) (Math.sqrt(random.nextFloat()) * 0.75);
                float theta = (float) (random.nextFloat() * Math.PI * 2.0);
                float px = (float) (r * Math.cos(theta));
                float py = (float) (r * Math.sin(theta));

                xArr[i] = px;
                yArr[i] = py;

                float orbital = 0.18f + 0.14f * random.nextFloat();
                vxArr[i] = -py * orbital;
                vyArr[i] = px * orbital;

                mArr[i] = 0.7f + 1.6f * random.nextFloat();
            }

            for (int i = 0; i < count; i++) {
                int base = i * count;
                for (int j = 0; j < count; j++) {
                    maskArr[base + j] = i == j ? 0.0f : 1.0f;
                }
            }

            this.x = Tensor.of(xArr, vecShape);
            this.y = Tensor.of(yArr, vecShape);
            this.vx = Tensor.of(vxArr, vecShape);
            this.vy = Tensor.of(vyArr, vecShape);
            this.mass = Tensor.of(mArr, vecShape);
            this.pairMask = Tensor.of(maskArr, pairShape);
            this.onesCol = Tensor.ones(vecShape);
            this.onesRow = Tensor.ones(Shape.of(1, count));

            this.xHost = toHostVector(this.x);
            this.yHost = toHostVector(this.y);
            this.massHost = toHostVector(this.mass);

            // Pre-allocate arrays for mouse interaction
            this.mouseXArr = new float[count];
            this.mouseYArr = new float[count];
        }

        private void step(float mouseX, float mouseY, boolean mouseActive) {
            // Compute gravitational accelerations (n-body)
            Tensor xT = x.transpose(0, 1).contiguous();
            Tensor yT = y.transpose(0, 1).contiguous();
            Tensor mT = mass.transpose(0, 1).contiguous();

            Tensor xi = x.matmul(onesRow);
            Tensor yi = y.matmul(onesRow);
            Tensor xj = onesCol.matmul(xT);
            Tensor yj = onesCol.matmul(yT);

            Tensor dx = xj.subtract(xi);
            Tensor dy = yj.subtract(yi);

            Tensor dist2 = dx.square().add(dy.square()).add(softening);
            Tensor invDist = dist2.sqrt().reciprocal();
            Tensor invDist3 = invDist.square().multiply(invDist);

            Tensor mj = onesCol.matmul(mT);
            Tensor accelScale = invDist3.multiply(mj).multiply(pairMask).multiply(g);

            Tensor ax = dx.multiply(accelScale).matmul(onesCol);
            Tensor ay = dy.multiply(accelScale).matmul(onesCol);

            // Materialize accelerations before applying mouse forces
            // This prevents the mouse position from being embedded as constants
            float[] axHost = toHostVector(ax);
            float[] ayHost = toHostVector(ay);

            // Get current positions and velocities
            float[] xArr = toHostVector(x);
            float[] yArr = toHostVector(y);
            float[] vxArr = toHostVector(vx);
            float[] vyArr = toHostVector(vy);

            if (mouseActive) {
                // Apply mouse forces on the host side to avoid kernel recompilation
                for (int i = 0; i < count; i++) {
                    float mdx = mouseX - xArr[i];
                    float mdy = mouseY - yArr[i];
                    float md2 = mdx * mdx + mdy * mdy + softening * 420.0f;
                    float mInv = 1.0f / (float) Math.sqrt(md2);
                    float mouseScale = mInv * mouseG;

                    // Add mouse gravity and spring forces to acceleration
                    axHost[i] += mdx * mouseScale + mdx * mouseSpring;
                    ayHost[i] += mdy * mouseScale + mdy * mouseSpring;

                    // Apply steering forces directly to velocity
                    float targetVx = mdx * mouseSteer;
                    float targetVy = mdy * mouseSteer;
                    vxArr[i] += (targetVx - vxArr[i]) * dt * 1.8f;
                    vyArr[i] += (targetVy - vyArr[i]) * dt * 1.8f;
                }
            }

            // Update velocities and positions
            for (int i = 0; i < count; i++) {
                vxArr[i] += axHost[i] * dt;
                vyArr[i] += ayHost[i] * dt;
                vxArr[i] *= damping;
                vyArr[i] *= damping;
                xArr[i] += vxArr[i] * dt;
                yArr[i] += vyArr[i] * dt;
            }

            vx = Tensor.of(vxArr, vecShape);
            vy = Tensor.of(vyArr, vecShape);
            x = Tensor.of(xArr, vecShape);
            y = Tensor.of(yArr, vecShape);

            keepInBounds();
            xHost = toHostVector(x);
            yHost = toHostVector(y);
        }

        private void keepInBounds() {
            float[] xArr = toHostVector(x);
            float[] yArr = toHostVector(y);
            float[] vxArr = toHostVector(vx);
            float[] vyArr = toHostVector(vy);

            float max = worldRadius;
            float bounce = 0.88f;

            for (int i = 0; i < count; i++) {
                if (xArr[i] < -max) {
                    xArr[i] = -max;
                    vxArr[i] = -vxArr[i] * bounce;
                } else if (xArr[i] > max) {
                    xArr[i] = max;
                    vxArr[i] = -vxArr[i] * bounce;
                }

                if (yArr[i] < -max) {
                    yArr[i] = -max;
                    vyArr[i] = -vyArr[i] * bounce;
                } else if (yArr[i] > max) {
                    yArr[i] = max;
                    vyArr[i] = -vyArr[i] * bounce;
                }
            }

            x = Tensor.of(xArr, vecShape);
            y = Tensor.of(yArr, vecShape);
            vx = Tensor.of(vxArr, vecShape);
            vy = Tensor.of(vyArr, vecShape);
        }

        private static float[] toHostVector(Tensor tensor) {
            MemoryView<?> src = tensor.materialize();
            int size = Math.toIntExact(src.shape().size());
            float[] out = new float[size];

            @SuppressWarnings("unchecked")
            MemoryDomain<Object> srcDomain =
                    (MemoryDomain<Object>)
                            Environment.current().runtimeFor(src.memory().device()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryView<Object> srcView = (MemoryView<Object>) src;

            MemoryDomain<float[]> dstDomain = DomainFactory.ofFloats();
            MemoryView<float[]> dstView =
                    MemoryView.of(
                            MemoryFactory.ofFloats(out),
                            DataType.FP32,
                            Layout.rowMajor(src.shape()));

            MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
            return out;
        }
    }
}

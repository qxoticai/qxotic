package com.qxotic.jota.examples.demos;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Path;

public final class RayTracerFusedDemo {
    private static final String OUTPUT = "render.ppm";
    private static final int WIDTH = 960, HEIGHT = 540;
    private static final float EPS = 0.001f, HUGE = 1e20f;

    public static void main(String[] args) throws IOException {
        Device device = parseDevice(stringArg(args, "--device", "native"));
        boolean listDevices = boolFlag(args, "--list-devices");

        Environment global = Environment.global();
        if (listDevices) {
            printRuntimeDiagnostics(global);
            return;
        }
        if (!global.runtimes().hasRuntime(device)) {
            throw new IllegalStateException(unavailableDeviceMessage(global, device));
        }

        // Use scoped environment to properly select backend
        Environment env = new Environment(device, DataType.FP32, global.runtimes());
        long start = System.nanoTime();
        try {
            Environment.with(
                    env,
                    () -> {
                        try {
                            RenderResult result = render(WIDTH, HEIGHT, 3);
                            PpmWriter.write(
                                    Path.of(OUTPUT), WIDTH, HEIGHT, result.r, result.g, result.b);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        return null;
                    });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException) {
                throw (IOException) e.getCause();
            }
            throw e;
        }
        long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
        System.out.printf(
                "Wrote %s [%dx%d] on %s in %d ms%n",
                OUTPUT, WIDTH, HEIGHT, device.name(), elapsedMs);
    }

    private static RenderResult render(int w, int h, int bounces) {
        int n = w * h;
        Shape shape = Shape.of(n, 1);
        Tensor[] dirs = generateRays(w, h, shape);
        Tensor rdx = dirs[0], rdy = dirs[1], rdz = dirs[2];
        Tensor rox = broadcast(0f, shape),
                roy = broadcast(-0.4f, shape),
                roz = broadcast(2.2f, shape);
        Sphere[] spheres =
                new Sphere[] {
                    new Sphere(0, -1000.5f, -3, 1000, 0.45f, 0.75f, 0.85f, 0.3f),
                    new Sphere(-0.7f, 0.15f, -1.5f, 0.4f, 1, 0.2f, 0.1f, 0.25f),
                    new Sphere(0, 0.4f, -1.2f, 0.3f, 0.9f, 0.9f, 0.1f, 0.4f),
                    new Sphere(0.7f, 0.25f, -1.4f, 0.35f, 0.1f, 0.7f, 1, 0.5f)
                };
        Tensor one = broadcast(1f, shape),
                zero = broadcast(0f, shape),
                half = broadcast(0.5f, shape);
        Tensor eps = broadcast(EPS, shape),
                huge = broadcast(HUGE, shape),
                hugeHalf = broadcast(HUGE * 0.5f, shape);
        Tensor accR = zero,
                accG = zero,
                accB = zero,
                thrR = one,
                thrG = one,
                thrB = one,
                alive = one;
        for (int i = 0; i < bounces; i++) {
            Hit hit =
                    trace(
                            spheres, rox, roy, roz, rdx, rdy, rdz, one, zero, eps, huge, hugeHalf,
                            shape);
            Tensor hitF = hit.mask.cast(DataType.FP32).multiply(alive);
            Tensor missF = alive.multiply(one.subtract(hitF));
            Tensor skyT = rdy.multiply(half).add(half).max(zero).min(one);
            accR = accR.add(thrR.multiply(skyT.multiply(0.3f).add(0.4f)).multiply(missF));
            accG = accG.add(thrG.multiply(skyT.multiply(0.4f).add(0.7f)).multiply(missF));
            accB = accB.add(thrB.multiply(skyT.multiply(0.85f).add(0.15f)).multiply(missF));
            Tensor ldx = hit.x.subtract(zero),
                    ldy = hit.y.subtract(broadcast(5f, shape)),
                    ldz = hit.z.subtract(zero);
            Tensor dist = ldx.square().add(ldy.square()).add(ldz.square()).add(0.1f).sqrt();
            ldx = ldx.divide(dist);
            ldy = ldy.divide(dist);
            ldz = ldz.divide(dist);
            Tensor ndotl =
                    hit.nx
                            .multiply(ldx)
                            .add(hit.ny.multiply(ldy))
                            .add(hit.nz.multiply(ldz))
                            .max(zero)
                            .min(one);
            Tensor shade = ndotl.multiply(0.7f).add(0.15f);
            Tensor local = hitF.multiply(one.subtract(hit.refl));
            accR = accR.add(thrR.multiply(hit.ar.multiply(shade)).multiply(local));
            accG = accG.add(thrG.multiply(hit.ag.multiply(shade)).multiply(local));
            accB = accB.add(thrB.multiply(hit.ab.multiply(shade)).multiply(local));
            thrR = thrR.multiply(hit.refl).multiply(hitF);
            thrG = thrG.multiply(hit.refl).multiply(hitF);
            thrB = thrB.multiply(hit.refl).multiply(hitF);
            alive = hitF;
            Tensor dot = rdx.multiply(hit.nx).add(rdy.multiply(hit.ny)).add(rdz.multiply(hit.nz));
            Tensor scale = dot.multiply(2f);
            rdx = rdx.subtract(hit.nx.multiply(scale));
            rdy = rdy.subtract(hit.ny.multiply(scale));
            rdz = rdz.subtract(hit.nz.multiply(scale));
            Tensor norm =
                    rdx.square().add(rdy.square()).add(rdz.square()).add(1e-6f).sqrt().reciprocal();
            rdx = rdx.multiply(norm);
            rdy = rdy.multiply(norm);
            rdz = rdz.multiply(norm);
            rox = hit.x.add(hit.nx.multiply(EPS * 4f));
            roy = hit.y.add(hit.ny.multiply(EPS * 4f));
            roz = hit.z.add(hit.nz.multiply(EPS * 4f));
        }
        return new RenderResult(
                toArray(accR.divide(accR.add(1f)).sqrt().max(zero).min(one)),
                toArray(accG.divide(accG.add(1f)).sqrt().max(zero).min(one)),
                toArray(accB.divide(accB.add(1f)).sqrt().max(zero).min(one)));
    }

    private static Tensor[] generateRays(int w, int h, Shape shape) {
        int n = w * h;
        float aspect = (float) w / h, fov = (float) Math.tan(Math.toRadians(52) * 0.5);
        float[] xs = new float[w], ys = new float[h];
        for (int i = 0; i < w; i++) xs[i] = 2f * ((i + 0.5f) / w) - 1f;
        for (int i = 0; i < h; i++) ys[i] = 1f - 2f * ((i + 0.5f) / h);
        Tensor px = Tensor.of(xs, Shape.of(1, w)).broadcast(Shape.of(h, w)).reshape(Shape.of(n, 1));
        Tensor py = Tensor.of(ys, Shape.of(h, 1)).broadcast(Shape.of(h, w)).reshape(Shape.of(n, 1));
        Tensor dx = px.multiply(aspect * fov), dy = py.multiply(fov), dz = broadcast(-1f, shape);
        Tensor inv = dx.square().add(dy.square()).add(dz.square()).sqrt().reciprocal();
        return new Tensor[] {dx.multiply(inv), dy.multiply(inv), dz.multiply(inv)};
    }

    private static Hit trace(
            Sphere[] spheres,
            Tensor ox,
            Tensor oy,
            Tensor oz,
            Tensor dx,
            Tensor dy,
            Tensor dz,
            Tensor one,
            Tensor zero,
            Tensor eps,
            Tensor huge,
            Tensor hugeHalf,
            Shape shape) {
        Tensor bestT = huge, bx = zero, by = zero, bz = zero, bnx = zero, bny = zero, bnz = zero;
        Tensor bar = zero, bag = zero, bab = zero, brefl = zero;
        for (Sphere s : spheres) {
            Tensor cx = broadcast(s.cx, shape),
                    cy = broadcast(s.cy, shape),
                    cz = broadcast(s.cz, shape),
                    r = broadcast(s.r, shape);
            Tensor ocX = ox.subtract(cx), ocY = oy.subtract(cy), ocZ = oz.subtract(cz);
            Tensor a = dx.square().add(dy.square()).add(dz.square());
            Tensor b = ocX.multiply(dx).add(ocY.multiply(dy)).add(ocZ.multiply(dz)).multiply(2f);
            Tensor c = ocX.square().add(ocY.square()).add(ocZ.square()).subtract(r.square());
            Tensor disc = b.square().subtract(a.multiply(c).multiply(4f));
            Tensor sqrt = disc.max(zero).sqrt();
            Tensor denom = a.multiply(2f);
            Tensor tNear = b.negate().subtract(sqrt).divide(denom);
            Tensor tFar = b.negate().add(sqrt).divide(denom);
            Tensor t = huge;
            Tensor near = disc.greaterThan(zero).logicalAnd(tNear.greaterThan(eps));
            t =
                    tNear.multiply(near.cast(DataType.FP32))
                            .add(t.multiply(one.subtract(near.cast(DataType.FP32))));
            Tensor far =
                    disc.greaterThan(zero)
                            .logicalAnd(tFar.greaterThan(eps))
                            .logicalAnd(tFar.lessThan(t));
            t =
                    tFar.multiply(far.cast(DataType.FP32))
                            .add(t.multiply(one.subtract(far.cast(DataType.FP32))));
            Tensor better = t.lessThan(bestT).cast(DataType.FP32), worse = one.subtract(better);
            bestT = t.multiply(better).add(bestT.multiply(worse));
            Tensor hx = ox.add(dx.multiply(t)),
                    hy = oy.add(dy.multiply(t)),
                    hz = oz.add(dz.multiply(t));
            Tensor nx = hx.subtract(cx).divide(r),
                    ny = hy.subtract(cy).divide(r),
                    nz = hz.subtract(cz).divide(r);
            bx = hx.multiply(better).add(bx.multiply(worse));
            by = hy.multiply(better).add(by.multiply(worse));
            bz = hz.multiply(better).add(bz.multiply(worse));
            bnx = nx.multiply(better).add(bnx.multiply(worse));
            bny = ny.multiply(better).add(bny.multiply(worse));
            bnz = nz.multiply(better).add(bnz.multiply(worse));
            bar = broadcast(s.ar, shape).multiply(better).add(bar.multiply(worse));
            bag = broadcast(s.ag, shape).multiply(better).add(bag.multiply(worse));
            bab = broadcast(s.ab, shape).multiply(better).add(bab.multiply(worse));
            brefl = broadcast(s.refl, shape).multiply(better).add(brefl.multiply(worse));
        }
        return new Hit(bestT.lessThan(hugeHalf), bx, by, bz, bnx, bny, bnz, bar, bag, bab, brefl);
    }

    private static float[] toArray(Tensor t) {
        MemoryView<?> view = t.materialize();
        float[] arr = new float[(int) view.shape().size()];
        @SuppressWarnings("unchecked")
        MemoryView<Object> src = (MemoryView<Object>) view;
        MemoryDomain.copy(
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain(),
                src,
                DomainFactory.ofFloats(),
                MemoryView.of(
                        MemoryFactory.ofFloats(arr), DataType.FP32, Layout.rowMajor(view.shape())));
        return arr;
    }

    private static Tensor broadcast(float v, Shape s) {
        return Tensor.broadcasted(v, s);
    }

    private record Sphere(
            float cx, float cy, float cz, float r, float ar, float ag, float ab, float refl) {}

    private record Hit(
            Tensor mask,
            Tensor x,
            Tensor y,
            Tensor z,
            Tensor nx,
            Tensor ny,
            Tensor nz,
            Tensor ar,
            Tensor ag,
            Tensor ab,
            Tensor refl) {}

    private record RenderResult(float[] r, float[] g, float[] b) {}

    private static String stringArg(String[] args, String key, String fallback) {
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith(key + "=")) return args[i].substring(key.length() + 1);
        }
        return fallback;
    }

    private static boolean boolFlag(String[] args, String key) {
        for (String arg : args) if (key.equals(arg)) return true;
        return false;
    }

    private static Device parseDevice(String value) {
        String n = value == null ? "native" : value.trim().toLowerCase();
        return switch (n) {
            case "native" -> Device.NATIVE;
            case "panama", "cpu" -> Device.PANAMA;
            case "c" -> Device.C;
            case "hip" -> Device.HIP;
            case "opencl", "cl" -> Device.OPENCL;
            case "mojo" -> Device.MOJO;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported --device: "
                                    + value
                                    + " (use native|panama|c|hip|opencl|mojo)");
        };
    }

    private static void printRuntimeDiagnostics(Environment env) {
        System.out.println("Runtime availability:");
        for (RuntimeDiagnostic d : env.runtimeDiagnostics()) {
            System.out.println(
                    "- "
                            + d.device().leafName()
                            + " ["
                            + d.probe().status().name().toLowerCase()
                            + "] "
                            + d.probe().message());
            if (d.probe().hint() != null) System.out.println("  hint: " + d.probe().hint());
        }
    }

    private static String unavailableDeviceMessage(Environment env, Device device) {
        StringBuilder sb =
                new StringBuilder(
                        "Requested device runtime is not available: " + device.name() + "\n");
        for (RuntimeDiagnostic d : env.runtimes().diagnosticsFor(device)) {
            sb.append("- ")
                    .append(d.probe().status().name().toLowerCase())
                    .append(": ")
                    .append(d.probe().message())
                    .append("\n");
            if (d.probe().hint() != null)
                sb.append("  hint: ").append(d.probe().hint()).append("\n");
        }
        sb.append("Try: --device native or run with --list-devices for diagnostics.");
        return sb.toString();
    }
}

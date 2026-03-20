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
import com.qxotic.jota.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Path;

public final class RayTracerPpmDemo {

    private static final int DEFAULT_WIDTH = 960;
    private static final int DEFAULT_HEIGHT = 540;
    private static final String DEFAULT_OUTPUT = "raytrace.ppm";

    private static final float EPS = 0.001f;
    private static final float HUGE = 1.0e20f;

    private RayTracerPpmDemo() {}

    public static void main(String[] args) throws IOException {
        int width = intArg(args, "--width", DEFAULT_WIDTH);
        int height = intArg(args, "--height", DEFAULT_HEIGHT);
        int bounces = intArg(args, "--bounces", 3);
        String output = stringArg(args, "--output", DEFAULT_OUTPUT);

        Environment global = Environment.global();
        if (DemoDevices.hasListDevicesFlag(args)) {
            System.out.println(DemoDevices.listDevices(global));
            return;
        }
        Device device = DemoDevices.resolveDevice(global, stringArg(args, "--device", null));

        try {
            Environment.configureGlobal(new Environment(device, DataType.FP32, global.runtimes()));
        } catch (IllegalStateException ignored) {
        }

        long start = System.nanoTime();
        RenderResult render = render(width, height, bounces);
        PpmWriter.write(Path.of(output), width, height, render.r, render.g, render.b);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000L;

        System.out.println(
                "Wrote " + output + " [" + width + "x" + height + "] on " + device.runtimeId());
        System.out.println("Render time: " + elapsedMs + " ms");
    }

    private static RenderResult render(int width, int height, int maxBounces) {
        int pixels = width * height;
        Shape pixShape = Shape.of(pixels, 1);

        float[] dirXArr = new float[pixels];
        float[] dirYArr = new float[pixels];
        float[] dirZArr = new float[pixels];

        float aspect = (float) width / (float) height;
        float tanHalfFov = (float) Math.tan(Math.toRadians(65.0) * 0.5);

        int k = 0;
        for (int y = 0; y < height; y++) {
            float py = 1.0f - 2.0f * ((y + 0.5f) / height);
            for (int x = 0; x < width; x++) {
                float px = 2.0f * ((x + 0.5f) / width) - 1.0f;
                float dx = px * aspect * tanHalfFov;
                float dy = py * tanHalfFov;
                float dz = -1.0f;
                float invLen = (float) (1.0 / Math.sqrt(dx * dx + dy * dy + dz * dz));
                dirXArr[k] = dx * invLen;
                dirYArr[k] = dy * invLen;
                dirZArr[k] = dz * invLen;
                k++;
            }
        }

        Tensor rayDirX = Tensor.of(dirXArr, pixShape);
        Tensor rayDirY = Tensor.of(dirYArr, pixShape);
        Tensor rayDirZ = Tensor.of(dirZArr, pixShape);

        Tensor rayOriginX = Tensor.broadcasted(0.0f, pixShape);
        Tensor rayOriginY = Tensor.broadcasted(0.05f, pixShape);
        Tensor rayOriginZ = Tensor.broadcasted(2.8f, pixShape);

        Sphere[] spheres = {
            new Sphere(
                    -0.95f, -0.2f, -1.8f, 0.45f, 0.95f, 0.45f, 0.35f, 0.08f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f),
            new Sphere(
                    0.0f, -0.3f, -2.7f, 0.7f, 0.35f, 0.9f, 0.45f, 0.28f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f),
            new Sphere(
                    1.05f, -0.2f, -1.8f, 0.45f, 0.35f, 0.55f, 0.95f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f),
            new Sphere(
                    -0.2f, 0.62f, -2.2f, 0.28f, 0.95f, 0.85f, 0.35f, 0.18f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f),
            new Sphere(
                    0.45f, 0.45f, -1.45f, 0.23f, 0.95f, 0.35f, 0.85f, 0.85f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f),
            new Sphere(
                    0.0f, -1001.0f, -2.2f, 1000.0f, 0.72f, 0.72f, 0.76f, 0.05f, 1.0f, 8.0f, 0.14f,
                    0.16f, 0.18f, 0.0f)
        };

        Tensor onePix = Tensor.broadcasted(1.0f, pixShape);
        Tensor zeroPix = Tensor.broadcasted(0.0f, pixShape);

        Tensor accumR = Tensor.broadcasted(0.0f, pixShape);
        Tensor accumG = Tensor.broadcasted(0.0f, pixShape);
        Tensor accumB = Tensor.broadcasted(0.0f, pixShape);
        Tensor throughputR = onePix;
        Tensor throughputG = onePix;
        Tensor throughputB = onePix;
        Tensor alive = onePix;

        int bounceCount = Math.max(1, maxBounces);
        for (int bounce = 0; bounce < bounceCount; bounce++) {
            Hit hit =
                    traceScene(
                            rayOriginX,
                            rayOriginY,
                            rayOriginZ,
                            rayDirX,
                            rayDirY,
                            rayDirZ,
                            spheres,
                            pixShape);
            Tensor hitMaskF = hit.hitMask.cast(DataType.FP32).multiply(alive);
            Tensor missMaskF = alive.multiply(onePix.subtract(hitMaskF));

            Tensor skyT = rayDirY.multiply(0.5f).add(0.5f);
            Tensor skyR = onePix.subtract(skyT).multiply(1.0f).add(skyT.multiply(0.5f));
            Tensor skyG = onePix.subtract(skyT).multiply(1.0f).add(skyT.multiply(0.7f));
            Tensor skyB = onePix.subtract(skyT).multiply(1.0f).add(skyT.multiply(0.98f));

            accumR = accumR.add(throughputR.multiply(skyR).multiply(missMaskF));
            accumG = accumG.add(throughputG.multiply(skyG).multiply(missMaskF));
            accumB = accumB.add(throughputB.multiply(skyB).multiply(missMaskF));

            Tensor hitX = rayOriginX.add(rayDirX.multiply(hit.t));
            Tensor hitY = rayOriginY.add(rayDirY.multiply(hit.t));
            Tensor hitZ = rayOriginZ.add(rayDirZ.multiply(hit.t));

            Tensor checkerWave =
                    hitX.multiply(hit.checkerScale)
                            .sin()
                            .multiply(hitZ.multiply(hit.checkerScale).sin());
            Tensor checkerMask =
                    checkerWave
                            .greaterThan(zeroPix)
                            .cast(DataType.FP32)
                            .multiply(hit.checkerStrength);
            Tensor invCheckerMask = onePix.subtract(checkerMask);
            Tensor baseR =
                    hit.albedoR.multiply(invCheckerMask).add(hit.checkerR.multiply(checkerMask));
            Tensor baseG =
                    hit.albedoG.multiply(invCheckerMask).add(hit.checkerG.multiply(checkerMask));
            Tensor baseB =
                    hit.albedoB.multiply(invCheckerMask).add(hit.checkerB.multiply(checkerMask));

            Tensor shade = Tensor.broadcasted(0.07f, pixShape);
            Tensor spec = Tensor.broadcasted(0.0f, pixShape);

            shade =
                    shade.add(
                            lightDiffuse(
                                    hitX, hitY, hitZ, hit.nx, hit.ny, hit.nz, -2.2f, 3.2f, 1.7f,
                                    9.0f, pixShape));
            shade =
                    shade.add(
                            lightDiffuse(
                                    hitX, hitY, hitZ, hit.nx, hit.ny, hit.nz, 2.8f, 1.7f, 2.2f,
                                    5.5f, pixShape));
            spec =
                    spec.add(
                            lightSpecular(
                                    hitX, hitY, hitZ, hit.nx, hit.ny, hit.nz, rayDirX, rayDirY,
                                    rayDirZ, -2.2f, 3.2f, 1.7f, 26.0f, pixShape));
            spec =
                    spec.add(
                            lightSpecular(
                                    hitX, hitY, hitZ, hit.nx, hit.ny, hit.nz, rayDirX, rayDirY,
                                    rayDirZ, 2.8f, 1.7f, 2.2f, 18.0f, pixShape));

            Tensor litR = baseR.multiply(shade).add(spec.multiply(0.35f)).add(hit.emission);
            Tensor litG = baseG.multiply(shade).add(spec.multiply(0.35f)).add(hit.emission);
            Tensor litB = baseB.multiply(shade).add(spec.multiply(0.35f)).add(hit.emission);

            Tensor nonReflect = onePix.subtract(hit.reflectivity);
            Tensor localMask = hitMaskF.multiply(nonReflect);
            accumR = accumR.add(throughputR.multiply(litR).multiply(localMask));
            accumG = accumG.add(throughputG.multiply(litG).multiply(localMask));
            accumB = accumB.add(throughputB.multiply(litB).multiply(localMask));

            throughputR = throughputR.multiply(hit.reflectivity).multiply(hitMaskF);
            throughputG = throughputG.multiply(hit.reflectivity).multiply(hitMaskF);
            throughputB = throughputB.multiply(hit.reflectivity).multiply(hitMaskF);
            alive = hitMaskF;

            Tensor dot =
                    rayDirX.multiply(hit.nx)
                            .add(rayDirY.multiply(hit.ny))
                            .add(rayDirZ.multiply(hit.nz));
            Tensor twoDot = dot.multiply(2.0f);
            Tensor reflX = rayDirX.subtract(hit.nx.multiply(twoDot));
            Tensor reflY = rayDirY.subtract(hit.ny.multiply(twoDot));
            Tensor reflZ = rayDirZ.subtract(hit.nz.multiply(twoDot));
            Tensor reflInvLen =
                    reflX.square()
                            .add(reflY.square())
                            .add(reflZ.square())
                            .add(1.0e-6f)
                            .sqrt()
                            .reciprocal();
            rayDirX = reflX.multiply(reflInvLen);
            rayDirY = reflY.multiply(reflInvLen);
            rayDirZ = reflZ.multiply(reflInvLen);

            rayOriginX = hitX.add(hit.nx.multiply(EPS * 4.0f));
            rayOriginY = hitY.add(hit.ny.multiply(EPS * 4.0f));
            rayOriginZ = hitZ.add(hit.nz.multiply(EPS * 4.0f));
        }

        Tensor outR = accumR.max(zeroPix).min(onePix);
        Tensor outG = accumG.max(zeroPix).min(onePix);
        Tensor outB = accumB.max(zeroPix).min(onePix);

        return new RenderResult(toHostVector(outR), toHostVector(outG), toHostVector(outB));
    }

    private static Tensor lightDiffuse(
            Tensor hitX,
            Tensor hitY,
            Tensor hitZ,
            Tensor nx,
            Tensor ny,
            Tensor nz,
            float lx,
            float ly,
            float lz,
            float intensity,
            Shape shape) {
        Tensor ldx = Tensor.broadcasted(lx, shape).subtract(hitX);
        Tensor ldy = Tensor.broadcasted(ly, shape).subtract(hitY);
        Tensor ldz = Tensor.broadcasted(lz, shape).subtract(hitZ);
        Tensor dist2 = ldx.square().add(ldy.square()).add(ldz.square()).add(0.05f);
        Tensor invDist = dist2.sqrt().reciprocal();
        ldx = ldx.multiply(invDist);
        ldy = ldy.multiply(invDist);
        ldz = ldz.multiply(invDist);
        Tensor ndotl = nx.multiply(ldx).add(ny.multiply(ldy)).add(nz.multiply(ldz));
        Tensor lambert = ndotl.max(Tensor.broadcasted(0.0f, shape));
        return lambert.multiply(invDist.square()).multiply(intensity);
    }

    private static Tensor lightSpecular(
            Tensor hitX,
            Tensor hitY,
            Tensor hitZ,
            Tensor nx,
            Tensor ny,
            Tensor nz,
            Tensor rayDirX,
            Tensor rayDirY,
            Tensor rayDirZ,
            float lx,
            float ly,
            float lz,
            float intensity,
            Shape shape) {
        Tensor ldx = Tensor.broadcasted(lx, shape).subtract(hitX);
        Tensor ldy = Tensor.broadcasted(ly, shape).subtract(hitY);
        Tensor ldz = Tensor.broadcasted(lz, shape).subtract(hitZ);
        Tensor invL =
                ldx.square().add(ldy.square()).add(ldz.square()).add(0.05f).sqrt().reciprocal();
        ldx = ldx.multiply(invL);
        ldy = ldy.multiply(invL);
        ldz = ldz.multiply(invL);

        Tensor vx = rayDirX.negate();
        Tensor vy = rayDirY.negate();
        Tensor vz = rayDirZ.negate();
        Tensor hx = ldx.add(vx);
        Tensor hy = ldy.add(vy);
        Tensor hz = ldz.add(vz);
        Tensor invH =
                hx.square().add(hy.square()).add(hz.square()).add(1.0e-6f).sqrt().reciprocal();
        hx = hx.multiply(invH);
        hy = hy.multiply(invH);
        hz = hz.multiply(invH);

        Tensor ndoth = nx.multiply(hx).add(ny.multiply(hy)).add(nz.multiply(hz));
        Tensor s = ndoth.max(Tensor.broadcasted(0.0f, shape));
        Tensor s2 = s.square();
        Tensor s4 = s2.square();
        Tensor s8 = s4.square();
        Tensor s16 = s8.square();
        Tensor s32 = s16.square();
        return s32.multiply(intensity);
    }

    private static Hit traceScene(
            Tensor originX,
            Tensor originY,
            Tensor originZ,
            Tensor dirX,
            Tensor dirY,
            Tensor dirZ,
            Sphere[] spheres,
            Shape shape) {
        Tensor one = Tensor.broadcasted(1.0f, shape);
        Tensor zero = Tensor.broadcasted(0.0f, shape);
        Tensor bestT = Tensor.broadcasted(HUGE, shape);
        Tensor bestNx = zero;
        Tensor bestNy = zero;
        Tensor bestNz = zero;
        Tensor bestAlbedoR = zero;
        Tensor bestAlbedoG = zero;
        Tensor bestAlbedoB = zero;
        Tensor bestReflect = zero;
        Tensor bestEmission = zero;
        Tensor bestCheckerStrength = zero;
        Tensor bestCheckerScale = zero;
        Tensor bestCheckerR = zero;
        Tensor bestCheckerG = zero;
        Tensor bestCheckerB = zero;

        Tensor a = dirX.square().add(dirY.square()).add(dirZ.square());
        Tensor epsTensor = Tensor.broadcasted(EPS, shape);

        for (Sphere s : spheres) {
            Tensor ocx = originX.subtract(s.cx);
            Tensor ocy = originY.subtract(s.cy);
            Tensor ocz = originZ.subtract(s.cz);

            Tensor b =
                    ocx.multiply(dirX)
                            .add(ocy.multiply(dirY))
                            .add(ocz.multiply(dirZ))
                            .multiply(2.0f);
            Tensor c =
                    ocx.square().add(ocy.square()).add(ocz.square()).subtract(s.radius * s.radius);
            Tensor disc = b.square().subtract(a.multiply(c).multiply(4.0f));
            Tensor discPosMask = disc.greaterThan(zero);
            Tensor sqrtDisc = disc.max(zero).sqrt();

            Tensor denom = a.multiply(2.0f);
            Tensor tNear = b.negate().subtract(sqrtDisc).divide(denom);
            Tensor tFar = b.negate().add(sqrtDisc).divide(denom);

            Tensor tCandidate = Tensor.broadcasted(HUGE, shape);
            Tensor nearValid = discPosMask.logicalAnd(tNear.greaterThan(epsTensor));
            Tensor nearMask = nearValid.cast(DataType.FP32);
            tCandidate = tNear.multiply(nearMask).add(tCandidate.multiply(one.subtract(nearMask)));

            Tensor farValid = discPosMask.logicalAnd(tFar.greaterThan(epsTensor));
            Tensor farBetter = farValid.logicalAnd(tFar.lessThan(tCandidate));
            Tensor farMask = farBetter.cast(DataType.FP32);
            tCandidate = tFar.multiply(farMask).add(tCandidate.multiply(one.subtract(farMask)));

            Tensor better = tCandidate.lessThan(bestT).cast(DataType.FP32);
            Tensor invBetter = one.subtract(better);
            bestT = tCandidate.multiply(better).add(bestT.multiply(invBetter));

            Tensor hx = originX.add(dirX.multiply(tCandidate));
            Tensor hy = originY.add(dirY.multiply(tCandidate));
            Tensor hz = originZ.add(dirZ.multiply(tCandidate));

            Tensor nx = hx.subtract(s.cx).divide(s.radius);
            Tensor ny = hy.subtract(s.cy).divide(s.radius);
            Tensor nz = hz.subtract(s.cz).divide(s.radius);

            bestNx = nx.multiply(better).add(bestNx.multiply(invBetter));
            bestNy = ny.multiply(better).add(bestNy.multiply(invBetter));
            bestNz = nz.multiply(better).add(bestNz.multiply(invBetter));
            bestAlbedoR =
                    Tensor.broadcasted(s.r, shape)
                            .multiply(better)
                            .add(bestAlbedoR.multiply(invBetter));
            bestAlbedoG =
                    Tensor.broadcasted(s.g, shape)
                            .multiply(better)
                            .add(bestAlbedoG.multiply(invBetter));
            bestAlbedoB =
                    Tensor.broadcasted(s.b, shape)
                            .multiply(better)
                            .add(bestAlbedoB.multiply(invBetter));
            bestReflect =
                    Tensor.broadcasted(s.reflectivity, shape)
                            .multiply(better)
                            .add(bestReflect.multiply(invBetter));
            bestEmission =
                    Tensor.broadcasted(s.emission, shape)
                            .multiply(better)
                            .add(bestEmission.multiply(invBetter));
            bestCheckerStrength =
                    Tensor.broadcasted(s.checkerStrength, shape)
                            .multiply(better)
                            .add(bestCheckerStrength.multiply(invBetter));
            bestCheckerScale =
                    Tensor.broadcasted(s.checkerScale, shape)
                            .multiply(better)
                            .add(bestCheckerScale.multiply(invBetter));
            bestCheckerR =
                    Tensor.broadcasted(s.checkerR, shape)
                            .multiply(better)
                            .add(bestCheckerR.multiply(invBetter));
            bestCheckerG =
                    Tensor.broadcasted(s.checkerG, shape)
                            .multiply(better)
                            .add(bestCheckerG.multiply(invBetter));
            bestCheckerB =
                    Tensor.broadcasted(s.checkerB, shape)
                            .multiply(better)
                            .add(bestCheckerB.multiply(invBetter));
        }

        Tensor hitMask = bestT.lessThan(Tensor.broadcasted(HUGE * 0.5f, shape));
        return new Hit(
                hitMask,
                bestT,
                bestNx,
                bestNy,
                bestNz,
                bestAlbedoR,
                bestAlbedoG,
                bestAlbedoB,
                bestReflect,
                bestEmission,
                bestCheckerStrength,
                bestCheckerScale,
                bestCheckerR,
                bestCheckerG,
                bestCheckerB);
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
                        MemoryFactory.ofFloats(out), DataType.FP32, Layout.rowMajor(src.shape()));

        MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
        return out;
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

    private record Sphere(
            float cx,
            float cy,
            float cz,
            float radius,
            float r,
            float g,
            float b,
            float reflectivity,
            float checkerStrength,
            float checkerScale,
            float checkerR,
            float checkerG,
            float checkerB,
            float emission) {}

    private record Hit(
            Tensor hitMask,
            Tensor t,
            Tensor nx,
            Tensor ny,
            Tensor nz,
            Tensor albedoR,
            Tensor albedoG,
            Tensor albedoB,
            Tensor reflectivity,
            Tensor emission,
            Tensor checkerStrength,
            Tensor checkerScale,
            Tensor checkerR,
            Tensor checkerG,
            Tensor checkerB) {}

    private record RenderResult(float[] r, float[] g, float[] b) {}
}

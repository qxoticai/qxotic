package com.qxotic.jinfer;

import java.util.Set;
import java.util.TreeSet;
import org.graalvm.nativeimage.ImageSingletons;
import org.graalvm.nativeimage.Platform;
import org.graalvm.nativeimage.hosted.Feature;

/**
 * Pins jinfer's Vector API width to the ISA a native image is built for. The jar's
 * native-image.properties registers it, so every image with jinfer-core on its class path runs it
 * and needs no flag. On the JVM nothing loads it.
 *
 * <p>The kernels fix their species when {@link Segments} initializes, at build time, and left alone
 * it would take the width the build machine prefers, not {@code -march}'s: an x86-64-v3 image built
 * on an AVX-512 host would carry 512-bit species and run them through the Vector API's fallback, a
 * hundred times slower. This sets {@code jinfer.vector.width} first: 512 with AVX-512, 256 with
 * AVX2, 128 otherwise. An explicit {@code -Djinfer.vector.width} still narrows it, within what the
 * target executes, and {@code 0} selects the scalar kernels. jam-vector and toknroll-vector pin
 * their own widths the same way.
 */
final class VectorWidth implements Feature {
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        Set<String> features = targetFeatures();
        int max = features.contains("AVX512F") ? 512 : features.contains("AVX2") ? 256 : 128;
        String requested = System.getProperty("jinfer.vector.width", "").strip();
        int bits = requested.isEmpty() ? max : Integer.parseInt(requested);
        if (bits > max)
            throw new IllegalArgumentException(
                    "jinfer.vector.width="
                            + bits
                            + " exceeds the "
                            + max
                            + "-bit vectors of the target ISA");
        System.setProperty("jinfer.vector.width", Integer.toString(bits));
        System.out.println(
                "jinfer: Vector API width " + bits + " bits for a " + max + "-bit target");
    }

    /**
     * The target's CPU features, from the builder's {@code CPUFeatureAccess}: an internal API, so
     * reached by reflection (the jar compiles against GraalVM's public SDK only), and a GraalVM
     * that moves it gets its default target's features (AVX2 on x86-64, NEON elsewhere) instead of
     * a failed build.
     */
    private static Set<String> targetFeatures() {
        Set<String> features = new TreeSet<>();
        try {
            Class<?> cpu = Class.forName("com.oracle.svm.core.CPUFeatureAccess");
            Object singleton = ImageSingletons.lookup(cpu);
            for (Object f : (Iterable<?>) cpu.getMethod("buildtimeCPUFeatures").invoke(singleton))
                features.add(((Enum<?>) f).name());
        } catch (ReflectiveOperationException | LinkageError | RuntimeException unreadable) {
            if (Platform.includedIn(Platform.AMD64.class)) features.addAll(Set.of("AVX", "AVX2"));
            System.out.println(
                    "jinfer: cannot read the target's CPU features ("
                            + unreadable
                            + "), assuming GraalVM's default target's");
        }
        return features;
    }
}

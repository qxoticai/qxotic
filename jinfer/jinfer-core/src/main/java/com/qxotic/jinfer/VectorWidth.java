package com.qxotic.jinfer;

import com.oracle.svm.core.CPUFeatureAccess;
import java.util.Set;
import java.util.TreeSet;
import org.graalvm.nativeimage.Platform;
import org.graalvm.nativeimage.hosted.Feature;

/**
 * Pins the kernels' Vector API width to the ISA a native image is built for. The jar's
 * native-image.properties registers it, so every image with jinfer-core on its class path runs it
 * and needs no flag. On the JVM nothing loads it.
 *
 * <p>The kernels fix their species at build time ({@link Segments}, and jam-vector's {@code
 * VectorSupport}), and left alone they would take the width the build machine prefers, not {@code
 * -march}'s: an x86-64-v3 image built on an AVX-512 host would carry 512-bit species and run them
 * through the Vector API's fallback, a hundred times slower. This sets {@code jinfer.vectorBitSize}
 * before any kernel class initializes, and jam's {@code jam.vector.width} with it, since jinfer's
 * kernels call jam's: 512 with AVX-512, 256 with AVX2, 128 otherwise. An explicit {@code
 * -Djinfer.vectorBitSize} still narrows both, within what the target executes, and {@code 0}
 * selects jinfer's scalar kernels. ponytail: an image of jam alone passes {@code
 * -Djam.vector.width}; jam-vector gets a feature of its own when such an image exists.
 */
final class VectorWidth implements Feature {
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        int max = targetBits();
        String requested = System.getProperty("jinfer.vectorBitSize", "").strip();
        int bits = requested.isEmpty() ? max : Integer.parseInt(requested);
        if (bits > max)
            throw new IllegalArgumentException(
                    "jinfer.vectorBitSize="
                            + bits
                            + " exceeds the "
                            + max
                            + "-bit vectors of the target ISA");
        System.setProperty("jinfer.vectorBitSize", Integer.toString(bits));
        System.setProperty("jam.vector.width", Integer.toString(bits == 0 ? max : bits));
        System.out.println(
                "jinfer: Vector API width " + bits + " bits for a " + max + "-bit target");
    }

    /**
     * The widest vector the target executes, from the builder's CPU features: an internal API, so a
     * GraalVM that moves it gets its default target's width (AVX2 on x86-64, NEON elsewhere)
     * instead of a failed build.
     */
    private static int targetBits() {
        try {
            Set<String> features = new TreeSet<>();
            for (Object f : CPUFeatureAccess.singleton().buildtimeCPUFeatures())
                features.add(((Enum<?>) f).name());
            return features.contains("AVX512F") ? 512 : features.contains("AVX2") ? 256 : 128;
        } catch (LinkageError | RuntimeException moved) {
            System.out.println(
                    "jinfer: cannot read the target's CPU features ("
                            + moved
                            + "), assuming GraalVM's default target's");
            return Platform.includedIn(Platform.AMD64.class) ? 256 : 128;
        }
    }
}

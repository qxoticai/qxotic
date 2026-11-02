package com.qxotic.toknroll.libtoknroll;

import com.oracle.svm.core.CPUFeatureAccess;
import java.util.Set;
import java.util.TreeSet;
import org.graalvm.nativeimage.Platform;
import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeSystemProperties;

/**
 * Fits toknroll-vector to the ISA the image is built for. The engine fixes its species when its
 * classes initialize, at build time, and left alone it would take the width the build machine
 * prefers: a library for x86-64-v3 built on an AVX-512 host would carry 512-bit species and emulate
 * every lane. This reads the target's CPU features from the image builder and sets {@code
 * toknroll.vector.width} before any engine class initializes: 512 with AVX-512BW (the split
 * compares bytes), 256 with AVX2, 128 otherwise (NEON, AVX). An explicit {@code
 * -Dtoknroll.vector.width} still wins, within what the target executes.
 *
 * <p>An x86 target without AVX gets no vector code for these operations at all, and the engine
 * would run several times slower than core's own encoder: there the image starts with {@code
 * toknroll.engine=none}, which a {@code -Dtoknroll.engine} at run time overrides.
 */
public final class VectorWidth implements Feature {
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        Set<String> features = new TreeSet<>();
        for (Object f : CPUFeatureAccess.singleton().buildtimeCPUFeatures())
            features.add(((Enum<?>) f).name());
        int max = features.contains("AVX512BW") ? 512 : features.contains("AVX2") ? 256 : 128;
        String requested = System.getProperty("toknroll.vector.width", "").strip();
        int bits = requested.isEmpty() ? max : Integer.parseInt(requested);
        if (bits > max)
            throw new IllegalArgumentException(
                    "toknroll.vector.width="
                            + bits
                            + " exceeds the "
                            + max
                            + "-bit vectors of the target ISA");
        System.setProperty("toknroll.vector.width", Integer.toString(bits));
        if (Platform.includedIn(Platform.AMD64.class) && !features.contains("AVX")) {
            RuntimeSystemProperties.register("toknroll.engine", "none");
            System.out.println(
                    "toknroll: no AVX on the target, core encodes (toknroll.engine=none)");
        } else {
            System.out.println(
                    "toknroll: Vector API width " + bits + " bits for a " + max + "-bit target");
        }
    }
}

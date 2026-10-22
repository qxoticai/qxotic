package com.qxotic.jinfer.libjinfer;

import com.oracle.svm.core.CPUFeatureAccess;
import java.util.Set;
import java.util.TreeSet;
import org.graalvm.nativeimage.hosted.Feature;

/**
 * Pins the Vector API width to the ISA the image is built for. The kernels fix their species at
 * build time (jinfer-core's {@code Segments}, jam-vector's {@code VectorSupport}), and the default
 * there, {@code VectorShape.preferredShape()}, answers for the machine running the build: an {@code
 * -march=x86-64-v3} image built on an AVX-512 host would carry 512-bit species and emulate every
 * lane, a hundred times slower. This reads the target's CPU features from the image builder and
 * sets the width before any kernel class initializes. {@code -Djinfer.vectorBitSize} still
 * overrides, within what the target can execute.
 */
public final class VectorWidth implements Feature {
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        Set<String> features = new TreeSet<>();
        for (Object f : CPUFeatureAccess.singleton().buildtimeCPUFeatures())
            features.add(((Enum<?>) f).name());
        int max = features.contains("AVX512F") ? 512 : features.contains("AVX2") ? 256 : 128;
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
}

package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.oracle.svm.shared.AlwaysInline;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorShuffle;
import org.junit.jupiter.api.Test;

/**
 * A kernel method that takes or returns a vector must carry {@link AlwaysInline}.
 *
 * <p>A vector value cannot cross a call boundary in a register. While the compiler inlines the
 * call, the value is scalar-replaced into a SIMD register and nothing is allocated; the moment an
 * inliner declines - GraalVM's AOT inliner does, once the calling kernel's body grows past its
 * budget - the vector is materialized, and then EVERY Vector API operation in the caller
 * de-intrinsifies into the generic per-lane path. It is all-or-nothing per method, so the result is
 * a cliff and not a slope: LFM2.5-2.6B-Q8_0 decode measured 0.48 t/s against 27.5 t/s on a GraalVM
 * CE 25 image, entirely on whether MatMul's leaves were inlined.
 *
 * <p>native-image binds {@code com.oracle.svm.shared.AlwaysInline} by name, and
 * jam/graalvm-annotations declares it, so the annotation costs no GraalVM dependency and HotSpot
 * never sees the type. It is deliberately preferred over {@code -H:DirectedInline}, which names
 * caller and callee and fails silently twice over: a rule that matches nothing is discarded without
 * a warning, and the option is single-valued, so one module's rules displace another's.
 *
 * <p>The law is structural rather than a benchmark because the failure is invisible: the output
 * stays correct, no build or test fails, and only a profile or a disassembly of the shipped binary
 * shows it. Adding a vector-typed helper without the annotation is what regresses it, so that is
 * exactly what this test refuses.
 *
 * @see VectorSpeciesConstantTest for the sibling law, that a species must not cross a call either
 */
final class InlinePinsTest {

    @Test
    void everyKernelMethodTakingOrReturningAVectorIsPinnedInline() {
        List<Class<?>> kernels = kernelClasses();
        assertFalse(
                kernels.isEmpty(),
                "found no kernel classes to check - this law must not pass vacuously");

        List<String> unpinned = new ArrayList<>();
        for (Class<?> kernel : kernels)
            for (Method method : kernel.getDeclaredMethods())
                if (carriesAVector(method) && !method.isAnnotationPresent(AlwaysInline.class))
                    unpinned.add(kernel.getSimpleName() + "." + method.getName());

        assertTrue(
                unpinned.isEmpty(),
                () ->
                        "these methods pass a vector across a call boundary without pinning the"
                            + " call inline, so an inliner that declines materializes the vector"
                            + " and the whole calling kernel drops to the per-lane path: "
                                + unpinned
                                + ". Add @AlwaysInline(\"why\"), or restructure so no vector"
                                + " crosses the call.");
    }

    /** True when any parameter or the return type carries a vector payload. */
    private static boolean carriesAVector(Method method) {
        if (isVector(method.getReturnType())) return true;
        for (Class<?> parameter : method.getParameterTypes()) if (isVector(parameter)) return true;
        return false;
    }

    private static boolean isVector(Class<?> type) {
        return Vector.class.isAssignableFrom(type)
                || VectorShuffle.class.isAssignableFrom(type)
                || VectorMask.class.isAssignableFrom(type);
    }

    /**
     * Every kernel class in this package, read off the build output rather than a hand-kept list: a
     * new kernel must be covered the day it is written, not the day someone remembers it.
     */
    private static List<Class<?>> kernelClasses() {
        Path root =
                Path.of("target", "classes", "com", "qxotic", "jinfer", "kernels").toAbsolutePath();
        if (!Files.isDirectory(root)) return List.of();
        List<Class<?>> classes = new ArrayList<>();
        try (Stream<Path> files = Files.list(root)) {
            for (Path file : files.sorted().toList()) {
                String name = file.getFileName().toString();
                if (!name.endsWith(".class") || name.contains("$")) continue;
                try {
                    classes.add(
                            Class.forName(
                                    "com.qxotic.jinfer.kernels."
                                            + name.substring(
                                                    0, name.length() - ".class".length())));
                } catch (ClassNotFoundException | NoClassDefFoundError skip) {
                    // not on this run's classpath (a backend-gated kernel); nothing to enforce
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return classes;
    }
}

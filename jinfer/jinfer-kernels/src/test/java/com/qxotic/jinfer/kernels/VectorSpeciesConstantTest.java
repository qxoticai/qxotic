package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Segments;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

/**
 * A vector species must reach a kernel as a build-time constant ({@link Segments#F_SPECIES}), never
 * as a method parameter.
 *
 * <p>A species that arrives as an argument is constant only while the JIT inlines the method. The
 * moment an unrelated caller makes the kernel hot enough to compile standalone, every {@code
 * FloatVector.broadcast(sp, ...)} in it de-intrinsifies into an allocating {@code
 * FloatSpecies.broadcastBits}, the vectors stop being scalar-replaced, and the loop silently
 * collapses. Measured on FlashAttention's pv tiles: one 512x512 image encode went from 11 MB and
 * 1.07 s to 162 GB and 5.3 s once a language-model prefill shared the same leaves - on Graal, and
 * always on C2. Performance that depends on an inlining decision another caller can spoil is not
 * performance, so this is a structural law and not a benchmark.
 */
final class VectorSpeciesConstantTest {

    @Test
    void noKernelTakesItsSpeciesAsAParameter() {
        List<String> offenders = new ArrayList<>();
        for (Class<?> kernel :
                List.of(
                        FlashAttention.class,
                        MatMul.class,
                        Convert.class,
                        Ops.class,
                        Norms.class,
                        Activations.class,
                        Convolutions.class)) {
            for (Method method : kernel.getDeclaredMethods()) {
                for (Class<?> parameter : method.getParameterTypes()) {
                    if (VectorSpecies.class.isAssignableFrom(parameter)) {
                        offenders.add(kernel.getSimpleName() + "." + method.getName());
                    }
                }
            }
        }
        assertTrue(
                offenders.isEmpty(),
                () ->
                        "these kernels take a VectorSpecies parameter, so their vector ops"
                                + " de-intrinsify whenever the JIT compiles them standalone - read"
                                + " Segments.F_SPECIES in the body instead: "
                                + offenders);
    }
}

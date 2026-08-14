package com.qxotic.jam;

import java.lang.foreign.MemorySegment;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;
import java.util.function.Function;

/**
 * jam - fast multithreaded CPU matmul ({@code R = W @ Aᵀ}) with quantized weights.
 *
 * <p>{@code JAM} is the matmul contract. Backends are discovered through {@link Provider}. Operands
 * are native {@link MemorySegment}s + byte offsets; an implementation is responsible for its own
 * bounds/liveness handling. Liveness contract: every implementation of {@link #mm} must keep each
 * operand segment reachable across the entire kernel (a trailing {@code
 * Reference.reachabilityFence} per operand) - kernels that address via raw pointers are invisible
 * to the GC, and an operand backed by an automatic arena could otherwise be unmapped mid-call.
 */
public interface JAM {

    /** A discoverable JAM implementation. */
    interface Provider {
        /** Stable user-facing id, e.g. {@code native}, {@code vector}, or {@code scalar}. */
        String id();

        /** Selection hint for consumers: the highest priority provider is preferred. */
        int priority();

        /** Whether this provider can create a backend in the current JVM/process. */
        boolean isAvailable();

        /**
         * Returns a usable backend. The returned backend may be shared; callers must handle {@link
         * JAM#EBUSY} from {@link JAM#mm} when a provider uses a single shared context.
         */
        JAM create();
    }

    /**
     * Available JAM providers, highest priority first.
     *
     * <p>A provider can be turned off from the command line: {@code -Djam.<id>.disabled=true}
     * (system property only; only {@code true} disables). The flag names a provider {@link
     * Provider#id() id} - an id no installed provider carries is a silent no-op, and disabling
     * every provider yields an empty list (consumers fall back to their own floor).
     */
    static List<Provider> providers() {
        return select(
                ServiceLoader.load(Provider.class).stream()
                        .map(ServiceLoader.Provider::get)
                        .toList(),
                System::getProperty);
    }

    /** The disabled filter + availability + priority sort, factored out for in-memory tests. */
    static List<Provider> select(List<Provider> discovered, Function<String, String> property) {
        return discovered.stream()
                .filter(p -> !Boolean.parseBoolean(property.apply("jam." + p.id() + ".disabled")))
                .filter(JAM::available)
                .sorted(Comparator.comparingInt(Provider::priority).reversed())
                .toList();
    }

    private static boolean available(Provider provider) {
        try {
            return provider.isAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    // ── supported weight dtype tags: numerically identical to GGML's ggml_type ──
    int F32 = 0,
            F16 = 1,
            BF16 = 30,
            Q4_0 = 2,
            Q8_0 = 8,
            Q4_K = 12,
            Q5_K = 13,
            Q6_K = 14,
            MXFP4 = 39,
            NVFP4 = 40,
            Q1_0 = 41;

    // ── jam_status ──
    int OK = 0, EINVAL = 1, EUNSUPPORTED = 2, EBUSY = 3;

    /**
     * {@code R = W @ Aᵀ}. Each operand is a native {@link MemorySegment} + BYTE offset; {@code
     * ldw/lda/ldr} are ELEMENT row strides; {@code wt/at/rt} the operand dtypes ({@code at}, {@code
     * rt} are {@code F32} today). Returns a jam_status ({@link #OK} / {@link #EINVAL} / {@link
     * #EUNSUPPORTED} / {@link #EBUSY}).
     */
    int mm(
            MemorySegment w,
            long wOff,
            int wt,
            int ldw,
            MemorySegment a,
            long aOff,
            int at,
            int lda,
            MemorySegment r,
            long rOff,
            int rt,
            int ldr,
            int m,
            int n,
            int k);

    /** Contiguous shortcut - offsets 0, strides {@code k/k/m}, F32 activations + result. */
    default int mm(MemorySegment w, MemorySegment a, MemorySegment r, int wt, int m, int n, int k) {
        return mm(w, 0, wt, k, a, 0, F32, k, r, 0, F32, m, m, n, k);
    }
}

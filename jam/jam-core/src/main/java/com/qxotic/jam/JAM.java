package com.qxotic.jam;

import java.lang.foreign.MemorySegment;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;
import java.util.function.IntConsumer;

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
         * Returns a usable backend whose parallel work runs on {@code parallel}. The returned
         * backend may be shared; a shared backend either serializes concurrent {@link JAM#mm} calls
         * internally or surfaces {@link JAM#EBUSY} (e.g. the {@code native} provider serializes by
         * default and surfaces EBUSY with {@code -Djam.native.serial=false}); callers must handle
         * EBUSY either way.
         */
        JAM create(Parallel parallel);

        /** A backend that runs on the calling thread only. */
        default JAM create() {
            return create(Parallel.INLINE);
        }
    }

    /**
     * The host's parallel loop, offered to every backend. The backends here own no threads: a
     * kernel asks for {@code count} tasks, the host runs them {@link #width()} at a time on
     * whatever threads it has, and the call returns when all are done; the {@code slot} handed to a
     * {@link Body} indexes per-participant scratch. A kernel calls this from the thread that called
     * {@link JAM#mm}, never from inside one of its own tasks, and a backend with no CPU work (a GPU
     * matmul) simply never calls it. A backend may run its own threads instead, under one rule:
     * while {@code mm} runs it uses at most {@code width()} cores and the host's workers are
     * quiescent, and when {@code mm} returns its own workers are quiescent - two pools spinning on
     * one machine measured worse in every configuration tried.
     */
    interface Parallel {
        /**
         * A job (or, under {@link #forLoop}, one iteration) and the slot of the participant running
         * it.
         */
        @FunctionalInterface
        interface Job {
            void run(int index, int slot);
        }

        /**
         * The primitive: runs {@code body.run(j, slot)} for every job {@code j < jobs}, each
         * claimed by one participant; returns when all are done. {@code slot} is in {@code [0,
         * width())}, the calling thread is slot 0, and no two jobs running at once share a slot - a
         * slot indexes per-participant scratch. A job is whatever the backend wants balanced: a
         * panel, a slice of a fan-out, a row group.
         */
        void run(int jobs, Job body);

        /** How many jobs run at once at most: the compute thread budget, fixed for a lifetime. */
        int width();

        /**
         * Sugar over {@link #run} for many cheap uniform iterations: the range is presented as
         * {@code 2 x width()} contiguous bands, one job each, so a participant streams one long
         * band. {@code body.run(i, slot)} for every {@code i < count}.
         */
        default void forLoop(int count, Job body) {
            int jobs = (int) Math.min(count, 2L * width());
            if (jobs <= 0) return;
            run(
                    jobs,
                    (job, slot) -> {
                        int lo = (int) ((long) count * job / jobs),
                                hi = (int) ((long) count * (job + 1) / jobs);
                        for (int i = lo; i < hi; i++) body.run(i, slot);
                    });
        }

        /** {@link #forLoop(int, Job)} without the slot. */
        default void forLoop(int count, IntConsumer body) {
            forLoop(count, (i, slot) -> body.accept(i));
        }

        /** The calling thread alone. */
        Parallel INLINE =
                new Parallel() {
                    @Override
                    public void run(int jobs, Job body) {
                        for (int j = 0; j < jobs; j++) body.run(j, 0);
                    }

                    @Override
                    public int width() {
                        return 1;
                    }
                };
    }

    /**
     * Available JAM providers, highest priority first.
     *
     * <p>Providers own no threads: their parallel work runs on the {@link Parallel} the host hands
     * to {@link Provider#create(Parallel)}.
     *
     * <p>A provider can be turned off from the command line: {@code -Djam.<id>.disabled=true}
     * (system property only; only {@code true} disables). The flag names a provider {@link
     * Provider#id() id} - an id no installed provider carries is a silent no-op, and disabling
     * every provider yields an empty list (consumers fall back to their own floor).
     *
     * <p>Disabled providers are dropped BEFORE {@link Provider#isAvailable()} is probed, so a
     * disabled backend disappears exactly as if it had never been discovered - no native library
     * load, no availability side effect.
     */
    static List<Provider> providers() {
        return ServiceLoader.load(Provider.class).stream()
                .map(ServiceLoader.Provider::get)
                .filter(p -> !isDisabled(p))
                .filter(JAM::available)
                .sorted(Comparator.comparingInt(Provider::priority).reversed())
                .toList();
    }

    private static boolean isDisabled(Provider provider) {
        return Boolean.parseBoolean(System.getProperty("jam." + provider.id() + ".disabled"));
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

    /**
     * Weight-dtype flag for {@link #mm}: the W bytes hold the backend's packed in-memory layout of
     * {@code wt & ~PACKED} (see {@link #packSize}). Never a wire format.
     */
    int PACKED = 0x100;

    // ── jam_status ──
    int OK = 0, EINVAL = 1, EUNSUPPORTED = 2, EBUSY = 3;

    /**
     * Byte size of the packed in-memory weight layout this backend wants for a {@code [m x k]}
     * weight of {@code dtype}, or 0 to keep the canonical bytes (dtype not packable, shape
     * unsupported, or no packed kernels on this hardware). The CALLER produces the packed bytes
     * once at load (layout: jam.h {@code JAM_PACK_ABI}), drops the canonical copy, and calls {@link
     * #mm} with {@code wt | PACKED} and {@code ldw == k}. Values are exactly the canonical dequant
     * - packing only reorders bytes and widens scales.
     */
    default long packSize(int dtype, int m, int k) {
        return 0;
    }

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

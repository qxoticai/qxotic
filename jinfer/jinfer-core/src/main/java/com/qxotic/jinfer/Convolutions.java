// Convolution kernels over channel-major activations — {@code rows[channel][time]}, one contiguous
// row per channel. Sits beside Norms and Activations: the tensor-level operation a convolutional
// model (vocoder, encoder, any TDNN) needs, so ports express layers rather than SIMD.
//
// The shape that matters here is narrow and long: a vocoder ends up 12 to 96 channels wide and
// ~100k samples long. Time is then the only axis worth vectorizing, and with it contiguous a
// convolution is an FMA sweep per tap — no im2col, nothing materialized.
package com.qxotic.jinfer;

import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class Convolutions {
    private Convolutions() {}

    private static final VectorSpecies<Float> SPECIES = FloatTensor.F_SPECIES;

    /**
     * Time samples per unit of work. Only parallel granularity now that a unit writes its output
     * once; the inputs a single vector step touches are a few KB, so they stay in L1 regardless.
     */
    private static final int TILE = 4096;

    /**
     * Opt-in shape census ({@code -Djinfer.convProfile}): tally every call by shape and dump it at
     * exit, ranked by share of total FLOPs. Exists because tuning {@link #TILE_CODE} against a
     * hand-written shape ladder guesses both the shapes and their weights, and guessed both wrong
     * for Inflect2 — the real ladder runs 3x longer per chunk and is dominated by dilation 1, which
     * inverted the answer. Feed the dump to {@code bench/ConvPeak.java} to benchmark what a model
     * actually runs, weighted how it runs it.
     *
     * <p>Off by default: one {@code static final} false, so the branch folds away. NOT available in
     * a native image — {@code com.qxotic.*} initializes at build time, so this freezes to the build
     * machine's value (see {@link #TILE_CODE}).
     */
    private static final boolean PROFILE = Boolean.getBoolean("jinfer.convProfile");

    private static final java.util.Map<String, long[]> CENSUS =
            PROFILE ? new java.util.concurrent.ConcurrentHashMap<>() : java.util.Map.of();

    static {
        if (PROFILE) Runtime.getRuntime().addShutdownHook(new Thread(Convolutions::dumpCensus));
    }

    private static void tally(int inChannels, int outChannels, int time, int kernel, int dilation) {
        String key =
                String.format(
                        "%4d->%4d ch  %7dt  k=%-2d d=%-2d",
                        inChannels, outChannels, time, kernel, dilation);
        long flops = 2L * outChannels * inChannels * kernel * time;
        CENSUS.compute(
                key,
                (k, cell) -> {
                    long[] counts = cell == null ? new long[2] : cell;
                    counts[0]++;
                    counts[1] += flops;
                    return counts;
                });
    }

    private static void dumpCensus() {
        long total = CENSUS.values().stream().mapToLong(cell -> cell[1]).sum();
        if (total == 0) return;
        System.err.printf(
                "%n[conv census] %d distinct shapes, %.2f GFLOP total%n",
                CENSUS.size(), total / 1e9);
        System.err.printf("%-34s %8s %12s %8s%n", "shape", "calls", "GFLOP", "share");
        CENSUS.entrySet().stream()
                .sorted((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1]))
                .forEach(
                        e ->
                                System.err.printf(
                                        "%-34s %8d %12.3f %7.1f%%%n",
                                        e.getKey(),
                                        e.getValue()[0],
                                        e.getValue()[1] / 1e9,
                                        100.0 * e.getValue()[1] / total));
    }

    /**
     * Dilated 1-D convolution, "same" padding: {@code out[oc][t] = bias[oc] + Σ_ic Σ_k
     * taps[oc][ic][k] * in[ic][t + k*dilation - pad]}.
     *
     * <p>{@code taps} is dequantized and laid out {@code [outChannel][inChannel][kernel]}; {@code
     * bias} may be null. {@code in} and {@code out} must not be the same tensor.
     *
     * <p>Each output vector accumulates every tap in a register and is stored once — the difference
     * that matters, since a per-tap sweep would read and write the whole output row {@code
     * inChannels * kernel} times over (132 times for a 12-channel, 11-tap layer).
     */
    public static void conv1dRows(
            F32FloatTensor in,
            int inChannels,
            F32FloatTensor out,
            int outChannels,
            int time,
            int kernel,
            int dilation,
            float[] taps,
            FloatTensor bias) {
        if (PROFILE) tally(inChannels, outChannels, time, kernel, dilation);
        int pad = ((kernel - 1) * dilation) / 2;
        int tapsPerOutput = kernel * inChannels;
        int tiles = (time + TILE - 1) / TILE;
        // Output channels are handled in groups so that each input vector loaded serves the whole
        // group: with one channel at a time the loop issues one load per multiply-add and stalls on
        // the load ports, which is most of what separates this from a tuned BLAS convolution.
        int groups = (outChannels + GROUP - 1) / GROUP;

        // Units write disjoint slices of `out` and only read `in`, so no synchronization is needed.
        Parallel.parallelFor(
                0,
                groups * tiles,
                unit -> {
                    int firstChannel = (unit / tiles) * GROUP;
                    int channels = Math.min(GROUP, outChannels - firstChannel);
                    int from = (unit % tiles) * TILE, to = Math.min(time, from + TILE);
                    if (from >= to) return;

                    // Every tap is in range exactly while t is at least pad from either end; the
                    // interior is therefore branch-free, and the two edges are a few samples wide.
                    int bodyFrom = Math.max(from, pad), bodyTo = Math.min(to, time - pad);
                    if (bodyTo < bodyFrom) bodyTo = bodyFrom;

                    if (channels == GROUP)
                        group(
                                in,
                                inChannels,
                                out,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapsPerOutput,
                                bias,
                                firstChannel,
                                bodyFrom,
                                bodyTo);
                    for (int oc = firstChannel; oc < firstChannel + channels; oc++) {
                        float biasValue = bias == null ? 0f : bias.getFloat(oc);
                        int tapRow = oc * tapsPerOutput;
                        long outRow = (long) oc * time;
                        if (channels != GROUP)
                            body(
                                    in,
                                    inChannels,
                                    out,
                                    time,
                                    kernel,
                                    dilation,
                                    pad,
                                    taps,
                                    tapRow,
                                    biasValue,
                                    outRow,
                                    bodyFrom,
                                    bodyTo);
                        edge(
                                in,
                                inChannels,
                                out,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapRow,
                                biasValue,
                                outRow,
                                from,
                                bodyFrom);
                        edge(
                                in,
                                inChannels,
                                out,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapRow,
                                biasValue,
                                outRow,
                                bodyTo,
                                to);
                    }
                });
    }

    /**
     * Output channels per group — four accumulators of one vector each, so 4 FMAs per input load.
     */
    private static final int GROUP = 4;

    /**
     * Register-tile shape for {@link #group}, {@code channels x vectorsOfTime}: how many time
     * vectors are kept in flight per output channel. {@code -Djinfer.convTile=4x2} or {@code 4x4}.
     *
     * <p>Deeper tiles issue fewer instructions per multiply-add — 4x1 spends 1 load + 4 broadcasts
     * on 4 FMAs, while 4x2 spends 2 + 4 on 8 — but only while the accumulators stay in registers.
     * The ceiling is not the ISA's 32 zmm: stock C2 and Graal allocate only zmm0-15, the same limit
     * that pins jam's Q8_0 gemm to a 3x2 tile (see VectorSupport.autoTileCode). 4x1 holds 6 live
     * vectors and 4x2 holds 11, both under it; 4x4 needs 21 and spills, so it is here to be
     * measured, not to be selected.
     *
     * <p>Default stays 4x1 — 4x4 measured slower on Zen 4 (avx512, 16 lanes), and 4x2 has not yet
     * been measured on an unloaded machine.
     */
    private static final int TILE_CODE =
            switch (System.getProperty("jinfer.convTile", "auto")) {
                case "4x2" -> 1;
                case "4x4" -> 2;
                default -> 0; // 4x1
            };

    /**
     * The interior for a full group of {@link #GROUP} output channels: one input vector is loaded
     * per (input channel, tap) and multiplied into every channel's accumulator, so the loop issues
     * {@code GROUP} multiply-adds per load instead of one.
     */
    private static void group(
            F32FloatTensor in,
            int inChannels,
            F32FloatTensor out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapsPerOutput,
            FloatTensor bias,
            int firstChannel,
            int from,
            int to) {
        if (!FloatTensor.USE_VECTOR_API) {
            for (int oc = firstChannel; oc < firstChannel + GROUP; oc++)
                body(
                        in,
                        inChannels,
                        out,
                        time,
                        kernel,
                        dilation,
                        pad,
                        taps,
                        oc * tapsPerOutput,
                        bias == null ? 0f : bias.getFloat(oc),
                        (long) oc * time,
                        from,
                        to);
            return;
        }
        int lanes = SPECIES.length();
        int row0 = firstChannel * tapsPerOutput;
        int row1 = row0 + tapsPerOutput, row2 = row1 + tapsPerOutput, row3 = row2 + tapsPerOutput;
        int t = from;
        // A deeper tile, when one is selected, takes as much of the span as it can; whatever it
        // leaves falls through to the one-vector loop below and then to body()'s scalar tail.
        if (TILE_CODE != 0)
            t =
                    tile(
                            in,
                            inChannels,
                            out,
                            time,
                            kernel,
                            dilation,
                            pad,
                            taps,
                            tapsPerOutput,
                            bias,
                            firstChannel,
                            t,
                            to);
        for (; t <= to - lanes; t += lanes) {
            FloatVector acc0 = FloatVector.broadcast(SPECIES, biasOf(bias, firstChannel));
            FloatVector acc1 = FloatVector.broadcast(SPECIES, biasOf(bias, firstChannel + 1));
            FloatVector acc2 = FloatVector.broadcast(SPECIES, biasOf(bias, firstChannel + 2));
            FloatVector acc3 = FloatVector.broadcast(SPECIES, biasOf(bias, firstChannel + 3));
            for (int ic = 0; ic < inChannels; ic++) {
                long inRow = (long) ic * time;
                int tap = ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    FloatVector x =
                            load(
                                    in,
                                    in.vbase
                                            + (inRow + t + (long) k * dilation - pad)
                                                    * Float.BYTES);
                    acc0 = FloatVector.broadcast(SPECIES, taps[row0 + tap + k]).fma(x, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, taps[row1 + tap + k]).fma(x, acc1);
                    acc2 = FloatVector.broadcast(SPECIES, taps[row2 + tap + k]).fma(x, acc2);
                    acc3 = FloatVector.broadcast(SPECIES, taps[row3 + tap + k]).fma(x, acc3);
                }
            }
            long at = out.vbase + ((long) firstChannel * time + t) * Float.BYTES;
            store(out, at, acc0);
            store(out, at + (long) time * Float.BYTES, acc1);
            store(out, at + (long) 2 * time * Float.BYTES, acc2);
            store(out, at + (long) 3 * time * Float.BYTES, acc3);
        }
        for (int oc = firstChannel; oc < firstChannel + GROUP; oc++)
            body(
                    in,
                    inChannels,
                    out,
                    time,
                    kernel,
                    dilation,
                    pad,
                    taps,
                    oc * tapsPerOutput,
                    bias == null ? 0f : bias.getFloat(oc),
                    (long) oc * time,
                    t,
                    to);
    }

    /**
     * The deeper tiles {@link #TILE_CODE} can select: {@link #GROUP} output channels by two or four
     * vectors of time, each time vector loaded once and reused across every channel. Returns the
     * first time step it did not cover, for the caller's narrower loops to finish.
     */
    private static int tile(
            F32FloatTensor in,
            int inChannels,
            F32FloatTensor out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapsPerOutput,
            FloatTensor bias,
            int firstChannel,
            int from,
            int to) {
        int lanes = SPECIES.length();
        int row0 = firstChannel * tapsPerOutput;
        int row1 = row0 + tapsPerOutput, row2 = row1 + tapsPerOutput, row3 = row2 + tapsPerOutput;
        float bias0 = biasOf(bias, firstChannel), bias1 = biasOf(bias, firstChannel + 1);
        float bias2 = biasOf(bias, firstChannel + 2), bias3 = biasOf(bias, firstChannel + 3);
        long rowBytes = (long) time * Float.BYTES, laneBytes = (long) lanes * Float.BYTES;
        long outBase = out.vbase + (long) firstChannel * time * Float.BYTES;
        int t = from;
        if (TILE_CODE == 1) { // 4x2 — 11 live vectors, spill-free on a 16-register allocator
            for (int span = 2 * lanes; t <= to - span; t += span) {
                FloatVector a00 = FloatVector.broadcast(SPECIES, bias0), a01 = a00;
                FloatVector a10 = FloatVector.broadcast(SPECIES, bias1), a11 = a10;
                FloatVector a20 = FloatVector.broadcast(SPECIES, bias2), a21 = a20;
                FloatVector a30 = FloatVector.broadcast(SPECIES, bias3), a31 = a30;
                for (int ic = 0; ic < inChannels; ic++) {
                    long inRow = (long) ic * time;
                    int tap = ic * kernel;
                    for (int k = 0; k < kernel; k++) {
                        long at = in.vbase + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
                        FloatVector x0 = load(in, at), x1 = load(in, at + laneBytes);
                        FloatVector w = FloatVector.broadcast(SPECIES, taps[row0 + tap + k]);
                        a00 = w.fma(x0, a00);
                        a01 = w.fma(x1, a01);
                        w = FloatVector.broadcast(SPECIES, taps[row1 + tap + k]);
                        a10 = w.fma(x0, a10);
                        a11 = w.fma(x1, a11);
                        w = FloatVector.broadcast(SPECIES, taps[row2 + tap + k]);
                        a20 = w.fma(x0, a20);
                        a21 = w.fma(x1, a21);
                        w = FloatVector.broadcast(SPECIES, taps[row3 + tap + k]);
                        a30 = w.fma(x0, a30);
                        a31 = w.fma(x1, a31);
                    }
                }
                long at = outBase + (long) t * Float.BYTES;
                store(out, at, a00);
                store(out, at + laneBytes, a01);
                store(out, at + rowBytes, a10);
                store(out, at + rowBytes + laneBytes, a11);
                store(out, at + 2 * rowBytes, a20);
                store(out, at + 2 * rowBytes + laneBytes, a21);
                store(out, at + 3 * rowBytes, a30);
                store(out, at + 3 * rowBytes + laneBytes, a31);
            }
            return t;
        }
        for (int span = 4 * lanes; t <= to - span; t += span) { // 4x4 — 21 live vectors, spills
            FloatVector a00 = FloatVector.broadcast(SPECIES, bias0),
                    a01 = a00,
                    a02 = a00,
                    a03 = a00;
            FloatVector a10 = FloatVector.broadcast(SPECIES, bias1),
                    a11 = a10,
                    a12 = a10,
                    a13 = a10;
            FloatVector a20 = FloatVector.broadcast(SPECIES, bias2),
                    a21 = a20,
                    a22 = a20,
                    a23 = a20;
            FloatVector a30 = FloatVector.broadcast(SPECIES, bias3),
                    a31 = a30,
                    a32 = a30,
                    a33 = a30;
            for (int ic = 0; ic < inChannels; ic++) {
                long inRow = (long) ic * time;
                int tap = ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    long at = in.vbase + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
                    FloatVector x0 = load(in, at), x1 = load(in, at + laneBytes);
                    FloatVector x2 = load(in, at + 2 * laneBytes),
                            x3 = load(in, at + 3 * laneBytes);
                    FloatVector w = FloatVector.broadcast(SPECIES, taps[row0 + tap + k]);
                    a00 = w.fma(x0, a00);
                    a01 = w.fma(x1, a01);
                    a02 = w.fma(x2, a02);
                    a03 = w.fma(x3, a03);
                    w = FloatVector.broadcast(SPECIES, taps[row1 + tap + k]);
                    a10 = w.fma(x0, a10);
                    a11 = w.fma(x1, a11);
                    a12 = w.fma(x2, a12);
                    a13 = w.fma(x3, a13);
                    w = FloatVector.broadcast(SPECIES, taps[row2 + tap + k]);
                    a20 = w.fma(x0, a20);
                    a21 = w.fma(x1, a21);
                    a22 = w.fma(x2, a22);
                    a23 = w.fma(x3, a23);
                    w = FloatVector.broadcast(SPECIES, taps[row3 + tap + k]);
                    a30 = w.fma(x0, a30);
                    a31 = w.fma(x1, a31);
                    a32 = w.fma(x2, a32);
                    a33 = w.fma(x3, a33);
                }
            }
            long at = outBase + (long) t * Float.BYTES;
            store(out, at, a00);
            store(out, at + laneBytes, a01);
            store(out, at + 2 * laneBytes, a02);
            store(out, at + 3 * laneBytes, a03);
            store(out, at + rowBytes, a10);
            store(out, at + rowBytes + laneBytes, a11);
            store(out, at + rowBytes + 2 * laneBytes, a12);
            store(out, at + rowBytes + 3 * laneBytes, a13);
            store(out, at + 2 * rowBytes, a20);
            store(out, at + 2 * rowBytes + laneBytes, a21);
            store(out, at + 2 * rowBytes + 2 * laneBytes, a22);
            store(out, at + 2 * rowBytes + 3 * laneBytes, a23);
            store(out, at + 3 * rowBytes, a30);
            store(out, at + 3 * rowBytes + laneBytes, a31);
            store(out, at + 3 * rowBytes + 2 * laneBytes, a32);
            store(out, at + 3 * rowBytes + 3 * laneBytes, a33);
        }
        return t;
    }

    private static float biasOf(FloatTensor bias, int channel) {
        return bias == null ? 0f : bias.getFloat(channel);
    }

    /** The interior: every tap reads inside the sequence, so this carries no bounds tests. */
    private static void body(
            F32FloatTensor in,
            int inChannels,
            F32FloatTensor out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapRow,
            float bias,
            long outRow,
            int from,
            int to) {
        int lanes = SPECIES.length();
        int t = from;
        if (FloatTensor.USE_VECTOR_API) {
            // Four accumulators, not one: every tap of an output is a multiply-add into the same
            // register, so a single accumulator would serialize the whole (inChannel, tap) loop on
            // FMA latency. Four independent chains keep the units fed, and each broadcast tap is
            // reused across all four.
            int stride = 4 * lanes;
            for (; t <= to - stride; t += stride) {
                FloatVector acc0 = FloatVector.broadcast(SPECIES, bias);
                FloatVector acc1 = acc0, acc2 = acc0, acc3 = acc0;
                for (int ic = 0; ic < inChannels; ic++) {
                    long inRow = (long) ic * time;
                    int tap = tapRow + ic * kernel;
                    for (int k = 0; k < kernel; k++) {
                        FloatVector weight = FloatVector.broadcast(SPECIES, taps[tap + k]);
                        long at = in.vbase + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
                        acc0 = weight.fma(load(in, at), acc0);
                        acc1 = weight.fma(load(in, at + (long) lanes * Float.BYTES), acc1);
                        acc2 = weight.fma(load(in, at + (long) 2 * lanes * Float.BYTES), acc2);
                        acc3 = weight.fma(load(in, at + (long) 3 * lanes * Float.BYTES), acc3);
                    }
                }
                long at = out.vbase + (outRow + t) * Float.BYTES;
                store(out, at, acc0);
                store(out, at + (long) lanes * Float.BYTES, acc1);
                store(out, at + (long) 2 * lanes * Float.BYTES, acc2);
                store(out, at + (long) 3 * lanes * Float.BYTES, acc3);
            }
            for (; t <= to - lanes; t += lanes) {
                FloatVector acc = FloatVector.broadcast(SPECIES, bias);
                for (int ic = 0; ic < inChannels; ic++) {
                    long inRow = (long) ic * time;
                    int tap = tapRow + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                        acc =
                                FloatVector.broadcast(SPECIES, taps[tap + k])
                                        .fma(
                                                load(
                                                        in,
                                                        in.vbase
                                                                + (inRow
                                                                                + t
                                                                                + (long) k
                                                                                        * dilation
                                                                                - pad)
                                                                        * Float.BYTES),
                                                acc);
                }
                store(out, out.vbase + (outRow + t) * Float.BYTES, acc);
            }
        }
        for (; t < to; t++) {
            float acc = bias;
            for (int ic = 0; ic < inChannels; ic++) {
                long inRow = (long) ic * time;
                int tap = tapRow + ic * kernel;
                for (int k = 0; k < kernel; k++)
                    acc += taps[tap + k] * in.getFloat(inRow + t + (long) k * dilation - pad);
            }
            out.setFloat(outRow + t, acc);
        }
    }

    private static FloatVector load(F32FloatTensor tensor, long byteOffset) {
        return FloatVector.fromMemorySegment(
                SPECIES, tensor.vseg, byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    private static void store(F32FloatTensor tensor, long byteOffset, FloatVector value) {
        value.intoMemorySegment(tensor.vseg, byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    /** The first and last {@code pad} samples, where some taps fall outside the sequence. */
    private static void edge(
            F32FloatTensor in,
            int inChannels,
            F32FloatTensor out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapRow,
            float bias,
            long outRow,
            int from,
            int to) {
        for (int t = from; t < to; t++) {
            float acc = bias;
            for (int ic = 0; ic < inChannels; ic++) {
                long inRow = (long) ic * time;
                int tap = tapRow + ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    int source = t + k * dilation - pad;
                    if (source < 0 || source >= time) continue;
                    acc += taps[tap + k] * in.getFloat(inRow + source);
                }
            }
            out.setFloat(outRow + t, acc);
        }
    }
}

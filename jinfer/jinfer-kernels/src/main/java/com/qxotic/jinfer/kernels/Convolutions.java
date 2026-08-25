package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Segments;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Convolution kernels over dense FP32 views. */
public final class Convolutions {
    private Convolutions() {}

    private static final VectorSpecies<Float> SPECIES = Segments.F_SPECIES;

    /**
     * Time samples per unit of work. Only parallel granularity now that a unit writes its output
     * once; the inputs a single vector step touches are a few KB, so they stay in L1 regardless.
     */
    private static final int TILE = 4096;

    /**
     * 3x3 stride-2, pad-1 convolution over channel-major {@code [channel][time][frequency]} data.
     * Taps are {@code [outChannel][inChannel][3][3]}; output dimensions are ceilings of halves.
     * Input and output must not alias.
     */
    public static void conv2dStride2Pad1(
            MemoryView<MemorySegment> in,
            int timeIn,
            int frequencyIn,
            int inChannels,
            float[] taps,
            int outChannels,
            MemoryView<MemorySegment> out) {
        Raw x = Raw.f32(in, "in");
        Raw y = Raw.f32(out, "out");
        int timeOut = (timeIn - 1) / 2 + 1;
        int frequencyOut = (frequencyIn - 1) / 2 + 1;
        Parallel.forLoop(
                0,
                outChannels,
                oc -> {
                    for (int ot = 0; ot < timeOut; ot++) {
                        for (int of = 0; of < frequencyOut; of++) {
                            float sum = 0f;
                            for (int ic = 0; ic < inChannels; ic++) {
                                int tapBase = (oc * inChannels + ic) * 9;
                                for (int ky = 0; ky < 3; ky++) {
                                    int it = 2 * ot - 1 + ky;
                                    if (it < 0 || it >= timeIn) continue;
                                    for (int kx = 0; kx < 3; kx++) {
                                        int inf = 2 * of - 1 + kx;
                                        if (inf < 0 || inf >= frequencyIn) continue;
                                        sum +=
                                                taps[tapBase + ky * 3 + kx]
                                                        * readFloat(
                                                                x.vseg(),
                                                                x.vbase()
                                                                        + ((long) ic
                                                                                                * timeIn
                                                                                                * frequencyIn
                                                                                        + (long) it
                                                                                                * frequencyIn
                                                                                        + inf)
                                                                                * Float.BYTES);
                                    }
                                }
                            }
                            writeFloat(
                                    y.vseg(),
                                    y.vbase()
                                            + ((long) oc * timeOut * frequencyOut
                                                            + (long) ot * frequencyOut
                                                            + of)
                                                    * Float.BYTES,
                                    sum);
                        }
                    }
                });
    }

    /**
     * Left-padded depthwise convolution over time-major {@code [time][channel]} data. Taps are
     * {@code [channel][kernel]}, with the newest sample aligned to the last tap. Input and output
     * must not alias.
     */
    public static void causalDepthwise1d(
            MemoryView<MemorySegment> in,
            MemoryView<MemorySegment> taps,
            MemoryView<MemorySegment> out,
            int time,
            int channels,
            int kernel) {
        Raw x = Raw.f32(in, "in");
        Raw w = Raw.f32(taps, "taps");
        Raw y = Raw.f32(out, "out");
        Parallel.forLoop(
                time,
                t -> {
                    for (int c = 0; c < channels; c++) {
                        float sum = 0f;
                        for (int k = 0; k < kernel; k++) {
                            int source = t - kernel + 1 + k;
                            if (source < 0) continue;
                            sum +=
                                    readFloat(
                                                    w.vseg(),
                                                    w.vbase()
                                                            + ((long) c * kernel + k) * Float.BYTES)
                                            * readFloat(
                                                    x.vseg(),
                                                    x.vbase()
                                                            + ((long) source * channels + c)
                                                                    * Float.BYTES);
                        }
                        writeFloat(
                                y.vseg(), y.vbase() + ((long) t * channels + c) * Float.BYTES, sum);
                    }
                });
    }

    /**
     * Stateful causal depthwise convolution for Qwen3.5. History is {@code [kernel-1][channels]};
     * every output row is computed against the history as it stood at entry, then history is rolled
     * from the concatenation of old history and the complete input chunk.
     */
    public static void causalDepthwiseSilu(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> taps,
            MemoryView<MemorySegment> history,
            MemoryView<MemorySegment> output,
            int rows,
            int channels,
            int kernel) {
        causalDepthwiseSilu(input, taps, null, history, output, rows, channels, kernel);
    }

    /** Stateful causal depthwise convolution with an optional per-channel bias. */
    public static void causalDepthwiseSilu(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> taps,
            MemoryView<MemorySegment> bias,
            MemoryView<MemorySegment> history,
            MemoryView<MemorySegment> output,
            int rows,
            int channels,
            int kernel) {
        Raw x = Raw.f32(input, "input");
        Raw w = Raw.f32(taps, "taps");
        Raw b = bias == null ? null : Raw.f32(bias, "bias");
        Raw state = Raw.f32(history, "history");
        Raw out = Raw.f32(output, "output");
        int hist = kernel - 1;
        Parallel.forLoop(
                0,
                channels,
                c -> {
                    for (int row = 0; row < rows; row++) {
                        float sum =
                                b == null
                                        ? 0f
                                        : readFloat(b.vseg(), b.vbase() + (long) c * Float.BYTES);
                        for (int k = 0; k < kernel; k++) {
                            int pos = row - hist + k;
                            float value =
                                    pos < 0
                                            ? readFloat(
                                                    state.vseg(),
                                                    state.vbase()
                                                            + ((long) (pos + hist) * channels + c)
                                                                    * Float.BYTES)
                                            : readFloat(
                                                    x.vseg(),
                                                    x.vbase()
                                                            + ((long) pos * channels + c)
                                                                    * Float.BYTES);
                            sum +=
                                    readFloat(
                                                    w.vseg(),
                                                    w.vbase()
                                                            + ((long) c * kernel + k) * Float.BYTES)
                                            * value;
                        }
                        writeFloat(
                                out.vseg(),
                                out.vbase() + ((long) row * channels + c) * Float.BYTES,
                                sum * (1.0f / (1.0f + (float) Math.exp(-sum))));
                    }
                    for (int k = 0; k < hist; k++) {
                        int pos = rows - hist + k;
                        float value =
                                pos < 0
                                        ? readFloat(
                                                state.vseg(),
                                                state.vbase()
                                                        + ((long) (pos + hist) * channels + c)
                                                                * Float.BYTES)
                                        : readFloat(
                                                x.vseg(),
                                                x.vbase()
                                                        + ((long) pos * channels + c)
                                                                * Float.BYTES);
                        writeFloat(
                                state.vseg(),
                                state.vbase() + ((long) k * channels + c) * Float.BYTES,
                                value);
                    }
                });
    }

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

    private static final Map<String, long[]> CENSUS =
            PROFILE ? new ConcurrentHashMap<>() : Map.of();

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
     * bias} may be null. {@code in} and {@code out} must not be the same view.
     *
     * <p>Each output vector accumulates every tap in a register and is stored once — the difference
     * that matters, since a per-tap sweep would read and write the whole output row {@code
     * inChannels * kernel} times over (132 times for a 12-channel, 11-tap layer).
     */
    public static void conv1dRows(
            MemoryView<MemorySegment> in,
            int inChannels,
            MemoryView<MemorySegment> out,
            int outChannels,
            int time,
            int kernel,
            int dilation,
            float[] taps,
            MemoryView<MemorySegment> bias) {
        if (PROFILE) tally(inChannels, outChannels, time, kernel, dilation);
        Raw inRaw = Raw.f32(in, "in");
        Raw outRaw = Raw.f32(out, "out");
        Raw biasRaw = bias != null ? Raw.f32(bias, "bias") : null;
        int pad = ((kernel - 1) * dilation) / 2;
        int tapsPerOutput = kernel * inChannels;
        int tiles = (time + TILE - 1) / TILE;
        // Output channels are handled in groups so that each input vector loaded serves the whole
        // group: with one channel at a time the loop issues one load per multiply-add and stalls on
        // the load ports, which is most of what separates this from a tuned BLAS convolution.
        int groups = (outChannels + GROUP - 1) / GROUP;

        // Units write disjoint slices of `out` and only read `in`, so no synchronization is needed.
        Parallel.forLoop(
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
                                inRaw,
                                inChannels,
                                outRaw,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapsPerOutput,
                                biasRaw,
                                firstChannel,
                                bodyFrom,
                                bodyTo);
                    for (int oc = firstChannel; oc < firstChannel + channels; oc++) {
                        float biasValue = biasOf(biasRaw, oc);
                        int tapRow = oc * tapsPerOutput;
                        long outRow = (long) oc * time;
                        if (channels != GROUP)
                            body(
                                    inRaw,
                                    inChannels,
                                    outRaw,
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
                        // Clamp both edge spans to the tile's [from, to): when pad >= the tile
                        // span (pad > time is the extreme) bodyFrom overshoots `to`, and an
                        // unclamped edge would write outputs past the row - through GLOBAL_SEGMENT
                        // that is an unchecked wild write.
                        edge(
                                inRaw,
                                inChannels,
                                outRaw,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapRow,
                                biasValue,
                                outRow,
                                from,
                                Math.min(bodyFrom, to));
                        edge(
                                inRaw,
                                inChannels,
                                outRaw,
                                time,
                                kernel,
                                dilation,
                                pad,
                                taps,
                                tapRow,
                                biasValue,
                                outRow,
                                Math.max(bodyTo, from),
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
     * Live vectors are {@code channels*time + time + 1}: 4x1 holds 6, 4x2 11, 4x4 21, 6x4 29, 4x6
     * 31. Stock C2 and Graal allocate only zmm0-15, the same limit that pins jam's Q8_0 gemm to a
     * 3x2 tile (see VectorSupport.autoTileCode); a native image allocates from all 32 zmm, which is
     * why the ranking inverts between the two runtimes.
     *
     * <p>Measured on Zen 5 (9950X3D, avx512) against the Inflect2 census, idle machine:
     *
     * <ul>
     *   <li>NATIVE IMAGE — 4x4 wins by a wide margin: +14% (nano) / +29% (micro) end to end over
     *       4x1, and roughly half the convolution time. Deeper does NOT continue to pay: 4x6 (31
     *       live) and 6x4 (29 live) are 51-67% and 40-134% slower than 4x4 at every thread count,
     *       so the usable register ceiling is somewhere between 21 and 29 rather than at 32.
     *   <li>JIT — 4x1 is right. 4x2 is model- and JVM-dependent (nano regresses ~5% on GraalVM
     *       25.2.4 while micro gains ~5%), and 4x4 loses outright. Note the JIT is ~1.9x slower
     *       than the image on the same single-threaded convolutions either way.
     *   <li>Forcing {@code load}/{@code store} inline via hotspot_compile_commands made the JIT
     *       10-20% SLOWER (the loop is already register-starved) and did nothing at all in the
     *       image.
     * </ul>
     *
     * <p>So: default stays 4x1 for the JIT, and an image should be built with 4x4 — which it can
     * only be at BUILD time. {@code com.qxotic.*} initializes at build time, so this constant
     * freezes then and {@code -Djinfer.convTile} on a binary is silently ignored; pass it to the
     * image build instead (the {@code jinfer.convTile} pom property). Being a true constant is also
     * what lets the tile branch fold away entirely in the image.
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
            Raw in,
            int inChannels,
            Raw out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapsPerOutput,
            Raw bias,
            int firstChannel,
            int from,
            int to) {
        if (!Segments.USE_VECTOR_API) {
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
                        biasOf(bias, oc),
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
                                    in.vbase()
                                            + (inRow + t + (long) k * dilation - pad)
                                                    * Float.BYTES);
                    acc0 = FloatVector.broadcast(SPECIES, taps[row0 + tap + k]).fma(x, acc0);
                    acc1 = FloatVector.broadcast(SPECIES, taps[row1 + tap + k]).fma(x, acc1);
                    acc2 = FloatVector.broadcast(SPECIES, taps[row2 + tap + k]).fma(x, acc2);
                    acc3 = FloatVector.broadcast(SPECIES, taps[row3 + tap + k]).fma(x, acc3);
                }
            }
            long at = out.vbase() + ((long) firstChannel * time + t) * Float.BYTES;
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
                    biasOf(bias, oc),
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
            Raw in,
            int inChannels,
            Raw out,
            int time,
            int kernel,
            int dilation,
            int pad,
            float[] taps,
            int tapsPerOutput,
            Raw bias,
            int firstChannel,
            int from,
            int to) {
        int lanes = SPECIES.length();
        int row0 = firstChannel * tapsPerOutput;
        int row1 = row0 + tapsPerOutput, row2 = row1 + tapsPerOutput, row3 = row2 + tapsPerOutput;
        float bias0 = biasOf(bias, firstChannel), bias1 = biasOf(bias, firstChannel + 1);
        float bias2 = biasOf(bias, firstChannel + 2), bias3 = biasOf(bias, firstChannel + 3);
        long rowBytes = (long) time * Float.BYTES, laneBytes = (long) lanes * Float.BYTES;
        long outBase = out.vbase() + (long) firstChannel * time * Float.BYTES;
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
                        long at =
                                in.vbase() + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
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
                    long at = in.vbase() + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
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
                    a12 = w.fma(x1, a12);
                    a13 = w.fma(x1, a13);
                    w = FloatVector.broadcast(SPECIES, taps[row2 + tap + k]);
                    a20 = w.fma(x0, a20);
                    a21 = w.fma(x1, a21);
                    a22 = w.fma(x2, a22);
                    a23 = w.fma(x2, a23);
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

    private static float biasOf(Raw bias, int channel) {
        return bias == null
                ? 0f
                : readFloat(bias.vseg(), bias.vbase() + (long) channel * Float.BYTES);
    }

    /** The interior: every tap reads inside the sequence, so this carries no bounds tests. */
    private static void body(
            Raw in,
            int inChannels,
            Raw out,
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
        if (Segments.USE_VECTOR_API) {
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
                        long at =
                                in.vbase() + (inRow + t + (long) k * dilation - pad) * Float.BYTES;
                        acc0 = weight.fma(load(in, at), acc0);
                        acc1 = weight.fma(load(in, at + (long) lanes * Float.BYTES), acc1);
                        acc2 = weight.fma(load(in, at + (long) 2 * lanes * Float.BYTES), acc2);
                        acc3 = weight.fma(load(in, at + (long) 3 * lanes * Float.BYTES), acc3);
                    }
                }
                long at = out.vbase() + (outRow + t) * Float.BYTES;
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
                                                        in.vbase()
                                                                + (inRow
                                                                                + t
                                                                                + (long) k
                                                                                        * dilation
                                                                                - pad)
                                                                        * Float.BYTES),
                                                acc);
                }
                store(out, out.vbase() + (outRow + t) * Float.BYTES, acc);
            }
        }
        for (; t < to; t++) {
            float acc = bias;
            for (int ic = 0; ic < inChannels; ic++) {
                long inRow = (long) ic * time;
                int tap = tapRow + ic * kernel;
                for (int k = 0; k < kernel; k++)
                    acc +=
                            taps[tap + k]
                                    * readFloat(
                                            in.vseg(),
                                            in.vbase()
                                                    + (inRow + t + (long) k * dilation - pad)
                                                            * Float.BYTES);
            }
            writeFloat(out.vseg(), out.vbase() + (outRow + t) * Float.BYTES, acc);
        }
    }

    private static FloatVector load(Raw tensor, long byteOffset) {
        return FloatVector.fromMemorySegment(
                SPECIES, tensor.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    private static void store(Raw tensor, long byteOffset, FloatVector value) {
        value.intoMemorySegment(tensor.vseg(), byteOffset, ByteOrder.LITTLE_ENDIAN);
    }

    /** The first and last {@code pad} samples, where some taps fall outside the sequence. */
    private static void edge(
            Raw in,
            int inChannels,
            Raw out,
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
                    acc +=
                            taps[tap + k]
                                    * readFloat(
                                            in.vseg(), in.vbase() + (inRow + source) * Float.BYTES);
                }
            }
            writeFloat(out.vseg(), out.vbase() + (outRow + t) * Float.BYTES, acc);
        }
    }

    /**
     * Causal short-convolution as a dConv-tap FIR over bx = B∘x rows (ported verbatim from {@code
     * Lfm2.shortConvScan}). {@code packed} holds [seqLen][parts*dim] rows with B in block 0, the C
     * gate in block 1 and x in block 2; bx is materialized in place over block 0. For each channel:
     * {@code out[s] = C_gate[s] * (Σ_{k<hist} state[k]·kernel[k] + bx[s]·kernel[hist])}, where
     * {@code state} ([hist][dim], hist = dConv-1) holds the previous bx values; the newest bx rolls
     * into {@code state}. Per-channel taps at {@code kernel[c*dConv + k]}.
     */
    public static void shortConvScan(
            MemoryView<MemorySegment> kernel,
            MemoryView<MemorySegment> convState,
            MemoryView<MemorySegment> packed,
            MemoryView<MemorySegment> out,
            int seqLen,
            int dim,
            int dConv,
            int parts) {
        Raw kernelRaw = Raw.f32(kernel, "kernel");
        Raw convRaw = Raw.f32(convState, "convState");
        Raw tmpRaw = Raw.f32(packed, "packed");
        Raw outRaw = Raw.f32(out, "out");
        MemorySegment ks = kernelRaw.vseg(),
                cs = convRaw.vseg(),
                ts = tmpRaw.vseg(),
                os = outRaw.vseg();
        long kb = kernelRaw.vbase(), cb = convRaw.vbase(), tb = tmpRaw.vbase(), ob = outRaw.vbase();
        int hist = dConv - 1;
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * parts * dim, outOff = s * dim;
            for (int c = 0; c < dim; c++) {
                float b = readFloat(ts, tb + 4L * (tmpOff + c));
                float cg = readFloat(ts, tb + 4L * (tmpOff + dim + c));
                float xv = readFloat(ts, tb + 4L * (tmpOff + 2 * dim + c));
                float bx = b * xv;
                writeFloat(ts, tb + 4L * (tmpOff + c), bx);
                int kBase = c * dConv;
                float sum = 0f;
                for (int k = 0; k < hist; k++)
                    sum +=
                            readFloat(cs, cb + 4L * ((long) k * dim + c))
                                    * readFloat(ks, kb + 4L * (kBase + k));
                sum += bx * readFloat(ks, kb + 4L * (kBase + dConv - 1));
                writeFloat(os, ob + 4L * (outOff + c), cg * sum);
                for (int k = 0; k < hist - 1; k++)
                    writeFloat(
                            cs,
                            cb + 4L * ((long) k * dim + c),
                            readFloat(cs, cb + 4L * ((long) (k + 1) * dim + c)));
                if (hist > 0) writeFloat(cs, cb + 4L * ((long) (hist - 1) * dim + c), bx);
            }
        }
    }

    /**
     * The centered-window twin of {@link #shortConvScan} for segmented sequences (ported verbatim
     * from {@code Lfm2}'s segmented path): bx = B∘x is materialized in place over block 0 of {@code
     * packed} (block 1 = C gate, block 2 = x) FIRST, since the centered window reads neighbour
     * rows; then each segment {@code [segRow0[g], segRow0[g]+segLen[g])} is convolved with
     * "same"-padding that zeroes beyond the SEGMENT's own edges and gated: {@code out[s] =
     * C_gate[s] * Σ_k bx[s-pad+k]·kernel[k]}, pad = (dConv-1)/2.
     */
    public static void segmentedShortConv(
            MemoryView<MemorySegment> kernel,
            MemoryView<MemorySegment> packed,
            MemoryView<MemorySegment> out,
            int[] segRow0,
            int[] segLen,
            int seqLen,
            int dim,
            int dConv,
            int parts) {
        Raw kernelRaw = Raw.f32(kernel, "kernel");
        Raw tmp = Raw.f32(packed, "packed");
        Raw outRaw = Raw.f32(out, "out");
        int pad = (dConv - 1) / 2;
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * parts * dim;
            for (int c = 0; c < dim; c++) {
                writeFloat(
                        tmp.vseg(),
                        tmp.vbase() + (long) (tmpOff + c) * Float.BYTES,
                        readFloat(tmp.vseg(), tmp.vbase() + (long) (tmpOff + c) * Float.BYTES)
                                * readFloat(
                                        tmp.vseg(),
                                        tmp.vbase() + (long) (tmpOff + 2 * dim + c) * Float.BYTES));
            }
        }
        for (int g = 0; g < segLen.length; g++) {
            int r0 = segRow0[g], rEnd = r0 + segLen[g];
            for (int s = r0; s < rEnd; s++) {
                int tmpOff = s * parts * dim, outOff = s * dim;
                for (int c = 0; c < dim; c++) {
                    float cg =
                            readFloat(
                                    tmp.vseg(),
                                    tmp.vbase() + (long) (tmpOff + dim + c) * Float.BYTES);
                    int kBase = c * dConv;
                    float sum = 0f;
                    for (int k = 0; k < dConv; k++) {
                        int row = s - pad + k; // zero beyond this sequence's own edges
                        if (row >= r0 && row < rEnd) {
                            sum +=
                                    readFloat(
                                                    tmp.vseg(),
                                                    tmp.vbase()
                                                            + ((long) row * parts * dim + c)
                                                                    * Float.BYTES)
                                            * readFloat(
                                                    kernelRaw.vseg(),
                                                    kernelRaw.vbase()
                                                            + (long) (kBase + k) * Float.BYTES);
                        }
                    }
                    writeFloat(
                            outRaw.vseg(),
                            outRaw.vbase() + (long) (outOff + c) * Float.BYTES,
                            cg * sum);
                }
            }
        }
    }
}

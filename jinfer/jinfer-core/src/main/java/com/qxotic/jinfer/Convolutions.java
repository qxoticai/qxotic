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

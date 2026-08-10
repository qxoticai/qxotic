package com.qxotic.jinfer.x;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracle: ported x.Convolutions.conv1dRows vs jinfer-core on identical inputs — full
 * groups and partial groups, interior/body/edge spans, dilations, with and without bias.
 */
class ConvolutionsTest {

    private final Arena arena = Arena.ofAuto();

    private void parity(
            int inCh, int outCh, int time, int kernel, int dilation, boolean withBias, long seed) {
        Random rng = new Random(seed);
        int inN = inCh * time, outN = outCh * time;
        MemorySegment in = Oracles.f32(arena, inN, seed);
        MemorySegment outOld = arena.allocate(4L * outN, 64);
        MemorySegment outNew = arena.allocate(4L * outN, 64);
        float[] taps = new float[outCh * inCh * kernel];
        for (int i = 0; i < taps.length; i++) taps[i] = rng.nextFloat() * 2 - 1;
        MemorySegment bias = withBias ? Oracles.f32(arena, outCh, seed + 1) : null;

        FloatTensor biasOld = withBias ? Oracles.oldF32(bias, outCh) : null;
        MemoryView<MemorySegment> biasNew = withBias ? Oracles.f32View(bias, outCh) : null;
        com.qxotic.jinfer.Convolutions.conv1dRows(
                (F32FloatTensor) Oracles.oldF32(in, inN),
                inCh,
                (F32FloatTensor) Oracles.oldF32(outOld, outN),
                outCh,
                time,
                kernel,
                dilation,
                taps,
                biasOld);
        Convolutions.conv1dRows(
                Oracles.f32View(in, inN),
                inCh,
                Oracles.f32View(outNew, outN),
                outCh,
                time,
                kernel,
                dilation,
                taps,
                biasNew);
        Oracles.assertClose(
                outOld,
                outNew,
                outN,
                String.format(
                        "conv %d->%d t=%d k=%d d=%d bias=%s",
                        inCh, outCh, time, kernel, dilation, withBias),
                1e-4);
    }

    @Test
    void fullGroupsWithBias() {
        parity(8, 8, 500, 11, 1, true, 1);
    }

    @Test
    void partialGroupNoBias() {
        parity(3, 5, 200, 3, 2, false, 2);
    }

    @Test
    void singleChannel() {
        parity(1, 1, 100, 5, 1, true, 3);
    }

    @Test
    void dilatedWideKernel() {
        parity(12, 4, 96, 11, 4, true, 4);
    }

    @Test
    void timeSmallerThanPad() {
        parity(2, 6, 8, 11, 3, false, 5);
    }

    @Test
    void largerTileSpan() {
        parity(4, 8, 9000, 3, 1, true, 6);
    }
}

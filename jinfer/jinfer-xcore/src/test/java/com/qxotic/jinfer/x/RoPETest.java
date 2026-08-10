package com.qxotic.jinfer.x;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracle: ported x.RoPE vs jinfer-core RoPE on identical inputs. Pure scalar ops (mul,
 * add, cos, sin) are deterministic across compilation tiers, so ulp-bound equality is far above
 * noise — the shared {@link Oracles} helper is used for consistency.
 */
class RoPETest {

    private static final int HEAD_SIZE = 128;
    private static final int LANES = HEAD_SIZE / 2;
    private static final double THETA = 10000.0;

    private final Arena arena = Arena.ofAuto();

    private MemorySegment zeros(int n) {
        return arena.allocate(4L * n, 64);
    }

    private float get(MemorySegment seg, long i) {
        return seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i);
    }

    private void fillParity(
            int count,
            com.qxotic.jinfer.RoPE.Schedule oldSched,
            RoPE.Schedule newSched,
            String what) {
        int n = count * LANES;
        MemorySegment cosOld = zeros(n), sinOld = zeros(n);
        MemorySegment cosNew = zeros(n), sinNew = zeros(n);
        FloatTensor cosT = Oracles.oldF32(cosOld, n);
        FloatTensor sinT = Oracles.oldF32(sinOld, n);
        com.qxotic.jinfer.RoPE.fill(cosT, sinT, 3, count, LANES, oldSched);
        RoPE.fill(
                Oracles.f32View(cosNew, n), Oracles.f32View(sinNew, n), 3, count, LANES, newSched);
        Oracles.assertClose(cosOld, cosNew, n, what + " cos");
        Oracles.assertClose(sinOld, sinNew, n, what + " sin");
    }

    @Test
    void fillPlainParity() {
        fillParity(
                17,
                com.qxotic.jinfer.RoPE.plain(HEAD_SIZE, THETA),
                RoPE.plain(HEAD_SIZE, THETA),
                "plain");
    }

    @Test
    void fillWithFreqFactorsParity() {
        float[] factors = new float[LANES];
        Random rng = new Random(42);
        for (int j = 0; j < LANES; j++) {
            factors[j] = 0.5f + rng.nextFloat() * 8;
        }
        fillParity(
                13,
                com.qxotic.jinfer.RoPE.withFreqFactors(HEAD_SIZE, THETA, factors),
                RoPE.withFreqFactors(HEAD_SIZE, THETA, factors),
                "freqFactors");
    }

    @Test
    void fillYarnParity() {
        fillParity(
                11,
                com.qxotic.jinfer.RoPE.yarn(HEAD_SIZE, THETA, 8f, 8192, 32f, 1f, 1f, 0.1f),
                RoPE.yarn(HEAD_SIZE, THETA, 8f, 8192, 32f, 1f, 1f, 0.1f),
                "yarn");
    }

    @Test
    void fillPositionsArrayParity() {
        int count = 9;
        int n = count * LANES;
        int[] positions = {0, 5, 5, 6, 0, 1, 2, 100, 1000};
        MemorySegment cosOld = zeros(n), sinOld = zeros(n);
        MemorySegment cosNew = zeros(n), sinNew = zeros(n);
        FloatTensor cosT = Oracles.oldF32(cosOld, n);
        FloatTensor sinT = Oracles.oldF32(sinOld, n);
        com.qxotic.jinfer.RoPE.fill(
                cosT,
                sinT,
                positions,
                count,
                LANES,
                com.qxotic.jinfer.RoPE.plain(HEAD_SIZE, THETA));
        RoPE.fill(
                Oracles.f32View(cosNew, n),
                Oracles.f32View(sinNew, n),
                positions,
                count,
                LANES,
                RoPE.plain(HEAD_SIZE, THETA));
        Oracles.assertClose(cosOld, cosNew, n, "positions cos");
        Oracles.assertClose(sinOld, sinNew, n, "positions sin");
    }

    private void applyParity(boolean neox) {
        int heads = 4;
        int rows = 7;
        long qn = (long) heads * HEAD_SIZE;
        long tn = (long) rows * LANES;
        MemorySegment qOld = Oracles.f32(arena, (int) qn, 7);
        MemorySegment qNew = arena.allocate(4L * qn, 64);
        MemorySegment.copy(qOld, 0, qNew, 0, 4L * qn);
        MemorySegment cosSeg = Oracles.f32(arena, (int) tn, 8);
        MemorySegment sinSeg = Oracles.f32(arena, (int) tn, 9);
        FloatTensor qT = Oracles.oldF32(qOld, qn);
        FloatTensor cosT = Oracles.oldF32(cosSeg, tn);
        FloatTensor sinT = Oracles.oldF32(sinSeg, tn);
        MemoryView<MemorySegment> qV = Oracles.f32View(qNew, qn);
        MemoryView<MemorySegment> cosV = Oracles.f32View(cosSeg, tn);
        MemoryView<MemorySegment> sinV = Oracles.f32View(sinSeg, tn);
        for (int row = 0; row < rows; row++) {
            for (int h = 0; h < heads; h++) {
                long off = (long) h * HEAD_SIZE;
                if (neox) {
                    com.qxotic.jinfer.RoPE.applyNeox(qT, off, row, cosT, sinT, LANES);
                    RoPE.applyNeox(qV, off, row, cosV, sinV, LANES);
                } else {
                    com.qxotic.jinfer.RoPE.applyInterleaved(qT, off, row, cosT, sinT, LANES);
                    RoPE.applyInterleaved(qV, off, row, cosV, sinV, LANES);
                }
            }
        }
        Oracles.assertClose(qOld, qNew, (int) qn, neox ? "applyNeox" : "applyInterleaved");
    }

    @Test
    void applyInterleavedParity() {
        applyParity(false);
    }

    @Test
    void applyNeoxParity() {
        applyParity(true);
    }

    @Test
    void fillMatchesApplyLayout() {
        // fill + apply composed against old fill + old apply (NeoX), proving the ported pair
        // composes the same way, not just each piece in isolation
        int rows = 5;
        long qn = 2L * HEAD_SIZE;
        long tn = (long) rows * LANES;
        MemorySegment qOld = Oracles.f32(arena, (int) qn, 11);
        MemorySegment qNew = arena.allocate(4L * qn, 64);
        MemorySegment.copy(qOld, 0, qNew, 0, 4L * qn);
        MemorySegment cosOld = zeros((int) tn), sinOld = zeros((int) tn);
        MemorySegment cosNew = zeros((int) tn), sinNew = zeros((int) tn);
        FloatTensor qT = Oracles.oldF32(qOld, qn);
        FloatTensor cosTOld = Oracles.oldF32(cosOld, tn);
        FloatTensor sinTOld = Oracles.oldF32(sinOld, tn);
        com.qxotic.jinfer.RoPE.fill(
                cosTOld, sinTOld, 2, rows, LANES, com.qxotic.jinfer.RoPE.plain(HEAD_SIZE, THETA));
        RoPE.fill(
                Oracles.f32View(cosNew, tn),
                Oracles.f32View(sinNew, tn),
                2,
                rows,
                LANES,
                RoPE.plain(HEAD_SIZE, THETA));
        for (int row = 0; row < rows; row++) {
            com.qxotic.jinfer.RoPE.applyNeox(qT, HEAD_SIZE, row, cosTOld, sinTOld, LANES);
            RoPE.applyNeox(
                    Oracles.f32View(qNew, qn),
                    HEAD_SIZE,
                    row,
                    Oracles.f32View(cosNew, tn),
                    Oracles.f32View(sinNew, tn),
                    LANES);
        }
        Oracles.assertClose(qOld, qNew, (int) qn, "fill+apply composition");
    }

    @Test
    void rejectsWrongDtype() {
        MemorySegment seg = zeros(8);
        MemoryView<MemorySegment> f16 = Views.wrap(seg, DataType.FP16, Shape.flat(8L));
        MemoryView<MemorySegment> f32 = Views.wrap(seg, DataType.FP32, Shape.flat(8L));
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () -> RoPE.fill(f16, f32, 0, 1, 4, RoPE.plain(8, THETA)));
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class, () -> RoPE.applyNeox(f32, 0, 0, f16, f32, 4));
    }
}

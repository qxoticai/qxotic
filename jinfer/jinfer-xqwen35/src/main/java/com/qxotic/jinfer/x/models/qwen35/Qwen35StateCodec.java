package com.qxotic.jinfer.x.models.qwen35;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/**
 * Qwen3.5 resume-state codec: the full-attention layers store per-position K/V rows (full context,
 * linear position indexing - no ring); the gated-delta-net layers contribute their conv history
 * plus the S matrices ({@code heads x headVDim x headVDim} F32) as the block RESIDUE - ~66MB total
 * residue at 35B-A3B dims (30 linear layers plus conv), so this model is {@link
 * #coarseCheckpoints()}: cached prompts commit as ONE block per prompt, one residue per prompt
 * rather than one per turn. When present, the embedded MTP block contributes ordinary per-position
 * K/V rows plus one normalized target-hidden row in the residue; no separate cache format or mode
 * exists. MoE routing is per-token and carries no cross-token state; everything else is per-batch
 * scratch - which is why {@code State.reset} zeroes exactly the recurrent buffers, the MTP carry,
 * and the cursor.
 *
 * <p>Coarse consequences worth knowing: matching is ALL-OR-NOTHING - a request whose stream
 * diverges anywhere inside the defined span, or is shorter than it, restores nothing and
 * re-prefills silently; only the misses counter tells. The defined block is always PREFIX-ONLY
 * (define drops the trailing batch, or the trailing position of a single-batch prompt), so a
 * one-short serve can match it.
 */
final class Qwen35StateCodec implements StateCodec<Qwen35.State> {

    private final Qwen35.Configuration config;
    private final int recurrentFloats; // heads * headVDim * headVDim per linear layer
    private final int convFloats; // (ssmConvKernel - 1) * convChannels per linear layer
    private final long bytesPerPosition;
    private final long residueBytes;

    Qwen35StateCodec(Qwen35.Configuration config) {
        this.config = config;
        recurrentFloats =
                Math.multiplyExact(
                        config.ssmTimeStepRank(),
                        Math.multiplyExact(config.headVDim(), config.headVDim()));
        convFloats =
                Math.multiplyExact(Math.max(config.ssmConvKernel() - 1, 0), config.convChannels());
        long rowBytes = 0;
        long linearLayers = 0;
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isFullAttention()[layer]) {
                rowBytes =
                        Math.addExact(
                                rowBytes, Math.multiplyExact(2L * Short.BYTES, config.kvDim()));
            } else {
                linearLayers++;
            }
        }
        bytesPerPosition = rowBytes;
        residueBytes =
                Math.addExact(
                        Math.multiplyExact(
                                linearLayers,
                                Math.multiplyExact(
                                        (long) recurrentFloats + convFloats, Float.BYTES)),
                        config.hasMtp()
                                ? Math.multiplyExact((long) config.embeddingLength(), Float.BYTES)
                                : 0);
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    public boolean coarseCheckpoints() {
        return true;
    }

    @Override
    public void saveCheckpoint(Qwen35.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Qwen35.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Qwen35.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.kvDim());
        long elementOffset = Math.multiplyExact((long) from, config.kvDim());
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (!config.isFullAttention()[layer]) continue;
            offset +=
                    KvTransfer.transfer(
                            state.keyCache[layer], elementOffset, blob, offset, elements, save);
            offset +=
                    KvTransfer.transfer(
                            state.valueCache[layer], elementOffset, blob, offset, elements, save);
        }
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isFullAttention()[layer]) continue;
            offset +=
                    KvTransfer.transfer(
                            state.recurrentState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            recurrentFloats,
                            save);
            offset +=
                    KvTransfer.transfer(
                            state.convState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convFloats,
                            save);
        }
        if (config.hasMtp())
            KvTransfer.transfer(
                    state.pendingHidden,
                    DataType.FP32,
                    0,
                    blob,
                    offset,
                    config.embeddingLength(),
                    save);
    }
}

package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The {@link MediaProjector} contract, checked once per vision/audio port so no tower hand-rolls
 * its own harness and remembers only half of it. Asserted here:
 *
 * <ul>
 *   <li>{@code positions(media)} predicts the exact row count {@code project} emits;
 *   <li>every chunk is a dense, contiguous FP32 {@code [rows, modelDim]} segment view with finite
 *       values, alive inside the sink and EXPIRED once {@code project} returns (the scratch arena
 *       discipline callers rely on);
 *   <li>no chunk exceeds {@code maxChunkSize} rows, a non-positive {@code maxChunkSize} is
 *       rejected, and a {@code positions}-sized one is accepted;
 *   <li>projection is deterministic: two runs over the same media are bitwise identical.
 * </ul>
 *
 * <p>this checks the SHAPE of the contract, not the numbers - per-port numerics belong in the
 * ports' ComponentsTests against llama.cpp oracles. What it cannot check is geometry a toy tower
 * shares with itself (the Qwen3.5 merger bug hid behind projectorDim == visionDim); keep toy dims
 * ASYMMETRIC where the real model's are.
 */
public final class MediaProjectorContract {

    private MediaProjectorContract() {}

    public static <R extends Media> void assertContract(
            MediaProjector<R> projector, R media, int modelDim) {
        Objects.requireNonNull(projector, "projector");
        Objects.requireNonNull(media, "media");
        assertTrue(modelDim > 0, "modelDim must be positive");

        int positions = projector.positions(media);
        assertTrue(positions >= 1, "positions() must be >= 1 but was " + positions);
        assertEquals(projector.planId(), projector.planId(), "planId() must be stable");

        float[][] first = projectAndCheck(projector, media, positions, modelDim, true);
        float[][] second = projectAndCheck(projector, media, positions, modelDim, false);
        assertEquals(
                first.length, second.length, "chunk count must be stable across identical runs");
        for (int i = 0; i < first.length; i++) {
            assertArrayEquals(
                    first[i], second[i], 0f, "chunk " + i + " differs across identical runs");
        }

        assertThrows(
                IllegalArgumentException.class,
                () -> projector.project(media, 0, chunk -> {}),
                "maxChunkSize=0 must be rejected");
    }

    private static <R extends Media> float[][] projectAndCheck(
            MediaProjector<R> projector,
            R media,
            int positions,
            int modelDim,
            boolean checkExpiry) {
        List<float[]> chunks = new ArrayList<>();
        List<MemoryView<MemorySegment>> borrowed = new ArrayList<>();
        int[] rows = {0};
        projector.project(
                media,
                positions,
                chunk -> {
                    MemoryView<MemorySegment> view = Views.castToSegmentBacked(chunk, "chunk");
                    assertTrue(
                            view.memory().base().scope().isAlive(),
                            "chunk must be alive inside the sink");
                    Views.requireDense(view, DataType.FP32, "chunk");
                    assertEquals(2, view.shape().flatRank(), "chunk must be 2D [rows, modelDim]");
                    assertEquals(modelDim, view.shape().flatAt(1), "chunk modelDim");
                    long chunkRows = view.shape().flatAt(0);
                    assertTrue(chunkRows >= 1, "chunk must carry at least one row");
                    assertTrue(
                            chunkRows <= positions,
                            "chunk rows " + chunkRows + " exceed maxChunkSize " + positions);
                    float[] values = Views.toFloatArray(view, "chunk");
                    for (float value : values) {
                        assertTrue(Float.isFinite(value), "chunk must be finite");
                    }
                    chunks.add(values);
                    borrowed.add(view);
                    rows[0] += Math.toIntExact(chunkRows);
                });
        assertEquals(
                positions, rows[0], "project() must emit exactly the rows positions() promised");
        if (checkExpiry) {
            for (MemoryView<MemorySegment> view : borrowed) {
                assertFalse(
                        view.memory().base().scope().isAlive(),
                        "chunk views must expire when project() returns (scratch arena closed)");
            }
        }
        return chunks.toArray(new float[0][]);
    }
}

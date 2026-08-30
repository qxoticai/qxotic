package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class Qwen35PositionTest {

    @Test
    void multimodalPositionCompressionIsChunkAndCheckpointStable() {
        Qwen35.Configuration config = Qwen35MtpLoadTest.withoutMtp();
        try (Arena arena = Arena.ofConfined()) {
            Qwen35 model =
                    new Qwen35(
                            config,
                            null,
                            Qwen35.loadWeights(Qwen35MtpLoadTest.tensors(config, arena), config));
            MemoryView<MemorySegment> rows =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), 4, config.embeddingLength());
            Batch.Positions positions =
                    new Batch.Positions(3, new int[] {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1}, 2);

            try (Qwen35.State state = model.newState(8, 4)) {
                model.ingest(state, Batch.embeddings(rows, 4, false, null, positions));
                assertEquals(4, state.position());
                assertEquals(-2, state.ropeDelta);

                Qwen35CheckpointCodec codec = new Qwen35CheckpointCodec(config);
                MemorySegment checkpoint = arena.allocate(codec.byteSize(4), 64);
                codec.capture(state, 0, 4, checkpoint);
                state.reset();
                codec.restore(state, 0, 4, checkpoint);
                state.resumeAt(4);
                assertEquals(-2, state.ropeDelta);

                state.reset();
                model.ingest(
                        state,
                        Batch.embeddings(
                                rows.slice(0, 0, 2), 2, false, null, positions.slice(0, 2, false)));
                model.ingest(
                        state,
                        Batch.embeddings(
                                rows.slice(0, 2, 4), 2, false, null, positions.slice(2, 2, true)));
                assertEquals(4, state.position());
                assertEquals(-2, state.ropeDelta);
            }
        }
    }
}

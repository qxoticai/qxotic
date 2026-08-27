package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A state's recurrent memory (short-conv taps, delta-net state, the MTP pending hidden) must start
 * at zero however the arena behaves: the public newState(ctx, batch, arena) contract does not
 * promise zero-filled allocations.
 */
final class Qwen35DirtyArenaTest {

    /** Hands out memory pre-filled with 0xFF (NaN as floats), like a recycled malloc arena. */
    private static final class DirtyArena implements MemoryArena<MemorySegment> {
        private final MemoryArena<MemorySegment> inner;

        DirtyArena(MemoryArena<MemorySegment> inner) {
            this.inner = inner;
        }

        @Override
        public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
            Memory<MemorySegment> memory = inner.allocateMemory(byteSize, byteAlignment);
            memory.base().fill((byte) 0xFF);
            return memory;
        }

        @Override
        public Device device() {
            return inner.device();
        }

        @Override
        public long memoryGranularity() {
            return inner.memoryGranularity();
        }

        @Override
        public boolean isAlive() {
            return inner.isAlive();
        }

        @Override
        public void close() {
            inner.close();
        }
    }

    @Test
    @Tag("integration")
    void aFreshStateStartsFromZeroWhateverTheArenaHandsOut() throws Exception {
        Qwen35 model =
                Qwen35.loadModel(
                        TestModels.require("hf.co/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf"),
                        Arena.ofAuto());
        int vocab = model.configuration().vocabularySize();
        int[] prompt = model.tokenizer().encodeToArray("The capital of France is");
        try (Arena arena = Arena.ofConfined();
                Qwen35.State clean = model.newState(64, 8);
                Qwen35.State dirty =
                        model.newState(64, 8, new DirtyArena(MemoryAllocators.ofArena(arena)))) {
            model.ingest(clean, Batch.prefill(prompt));
            model.ingest(dirty, Batch.prefill(prompt));
            int expected =
                    Ops.argmax(Views.castToSegmentBacked(model.logits(clean, 0), "l"), 0, vocab);
            int actual =
                    Ops.argmax(Views.castToSegmentBacked(model.logits(dirty, 0), "l"), 0, vocab);
            assertEquals(expected, actual, "a dirty arena must not leak into the recurrent state");
        }
    }
}

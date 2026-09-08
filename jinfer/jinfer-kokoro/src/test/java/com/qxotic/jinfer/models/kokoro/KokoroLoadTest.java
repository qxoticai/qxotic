package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class KokoroLoadTest {

    @Test
    void loadsTheCanonicalModelAndVoice() throws Exception {
        var model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-f16.gguf");
        var voice =
                TestModels.require("simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf");

        Arena arena = Arena.ofShared();
        try {
            Kokoro kokoro = Kokoro.load(model, voice, arena);
            assertEquals(24_000, kokoro.configuration().sampleRate());
            assertEquals(510, kokoro.voice().maxPhonemes());
            assertEquals(459, kokoro.weights().model().size());
            assertTrue(kokoro.parameterCount() > 81_000_000);
            assertArrayEquals(new int[] {43, 5}, kokoro.symbols().toRaw("a!"));
            try (Kokoro.State state = kokoro.newState()) {
                float[] pcm = kokoro.synthesize(state, new int[] {43}, 1, 0);
                int allocations = state.scratchAllocations();
                assertArrayEquals(pcm, kokoro.synthesize(state, new int[] {43}, 1, 0));
                assertEquals(allocations, state.scratchAllocations());
                assertTrue(pcm.length > 0);
                assertEquals(0, pcm.length % 600);
                for (float value : pcm) {
                    assertTrue(Float.isFinite(value));
                    assertTrue(value >= -1 && value <= 1);
                }
            }
            try (Kokoro.State state = kokoro.newState()) {
                arena.close();
                assertThrows(
                        IllegalStateException.class,
                        () -> kokoro.synthesize(state, new int[] {43}, 1, 0));
            }
        } finally {
            if (arena.scope().isAlive()) arena.close();
        }
    }
}

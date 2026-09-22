package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryView;
import com.sun.management.ThreadMXBean;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The state-owned workspace: a warm state transcribes without allocating, and recycled buffers,
 * which come back holding the previous window, never leak into a result.
 */
@Tag("integration")
class ParakeetWorkspaceTest {

    private static Arena arena;
    private static Parakeet parakeet;

    @BeforeAll
    static void load() throws IOException {
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");
        arena = Arena.ofShared();
        parakeet = Fixtures.load(model, arena);
    }

    @AfterAll
    static void unload() {
        if (arena != null) arena.close();
    }

    @Test
    void aWarmStateAllocatesNothing() {
        float[] longer = signal(12, 1), shorter = signal(5, 2);
        try (Parakeet.State state = parakeet.newState()) {
            parakeet.transcribe(state, longer);
            int warm = state.scratchAllocations();
            assertTrue(warm > 0, "the first window fills the workspace");
            parakeet.transcribe(state, longer);
            parakeet.transcribe(state, shorter); // smaller windows reuse the larger buffers
            assertEquals(warm, state.scratchAllocations());

            // ...and nothing sidesteps the workspace. On the calling thread a cold 12 s window
            // allocated ~120 MB (the subsampling array alone is ~39 MB); a warm one allocates
            // 6.2 MB, stable to a few KB (mel features, the stream's PCM buffer, Vector API
            // boxes). One per-layer [frames][dim] array coming back would add ~15 MB.
            long before = allocatedBytes();
            parakeet.transcribe(state, longer);
            long used = allocatedBytes() - before;
            assertTrue(used < 10_000_000, "warm transcription allocated " + used + " bytes");
        }
        // Windows of different lengths (committed windows, then the tail) share one workspace too.
        try (var chunks = Fixtures.chunkSeconds(10)) {
            try (Parakeet.State state = parakeet.newState()) {
                float[] windowed = signal(27, 3);
                parakeet.transcribe(state, windowed);
                int warm = state.scratchAllocations();
                parakeet.transcribe(state, windowed);
                assertEquals(warm, state.scratchAllocations());
            }
        }
    }

    /**
     * Every buffer a NaN window touches is left NaN; the next window on the same workspace must
     * overwrite all of it before reading, so its output matches a fresh workspace's. The poison
     * pass runs the pipeline's whole order (encode, joint projection, decode), so every slot the
     * clip reuses starts dirty.
     */
    @Test
    void staleBuffersNeverLeakIntoTheNextWindow() {
        float[] clip = signal(6, 4);
        float[] poison = new float[parakeet.sampleRate() * 9]; // longer: covers the clip's buffers
        Arrays.fill(poison, Float.NaN);
        Pass expected = Fixtures.onScratch(workspace -> pass(clip, workspace));
        Pass actual =
                Fixtures.onScratch(
                        workspace -> {
                            pass(poison, workspace);
                            workspace.rewind();
                            return pass(clip, workspace);
                        });
        assertClose(expected.encoded(), actual.encoded(), "encoder output");
        assertClose(expected.joint(), actual.joint(), "joint projection");
        assertEquals(expected.tokens(), actual.tokens(), "decoded tokens");
    }

    private record Pass(float[] encoded, float[] joint, List<Integer> tokens) {}

    /** One window, in the pipeline's allocation order; results copied out of the workspace. */
    private static Pass pass(float[] pcm, Workspace workspace) {
        ParakeetEncoder encoder = parakeet.weights().encoder();
        ParakeetTdt decoder = parakeet.weights().decoder();
        int frames = encoder.frames(pcm.length);
        MemoryView<MemorySegment> encoded = encoder.encode(pcm, 0, pcm.length, null, workspace);
        float[] joint = decoder.encProjection(encoded, frames, workspace);
        List<Integer> tokens =
                decoder.decode(joint, frames, workspace).stream()
                        .map(ParakeetTdt.Emission::token)
                        .toList();
        int jointSize = frames * decoder.config().jointHidden();
        return new Pass(
                Views.toFloatArray(encoded, "encoded"), Arrays.copyOf(joint, jointSize), tokens);
    }

    private static long allocatedBytes() {
        return ((ThreadMXBean) ManagementFactory.getThreadMXBean())
                .getThreadAllocatedBytes(Thread.currentThread().threadId());
    }

    /** Finite everywhere, and equal up to the engine's run-to-run float noise. */
    private static void assertClose(float[] expected, float[] actual, String what) {
        assertEquals(expected.length, actual.length, what + " length");
        double error = 0, norm = 0;
        for (int i = 0; i < expected.length; i++) {
            assertTrue(Float.isFinite(actual[i]), what + ": non-finite at " + i);
            error += Math.pow(actual[i] - expected[i], 2);
            norm += Math.pow(expected[i], 2);
        }
        assertTrue(
                Math.sqrt(error / norm) < 1e-3, what + ": relative L2 " + Math.sqrt(error / norm));
    }

    /** Deterministic speech-band test audio: gliding tones over light noise. */
    private static float[] signal(int seconds, long seed) {
        Random random = new Random(seed);
        float[] pcm = new float[16_000 * seconds];
        double phase = 0;
        for (int i = 0; i < pcm.length; i++) {
            double t = i / 16_000.0;
            phase += 2 * Math.PI * (300 + 200 * Math.sin(2 * Math.PI * 0.7 * t)) / 16_000;
            pcm[i] =
                    (float)
                            (0.3 * Math.sin(phase) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t))
                                    + 0.02 * random.nextGaussian());
        }
        return pcm;
    }
}

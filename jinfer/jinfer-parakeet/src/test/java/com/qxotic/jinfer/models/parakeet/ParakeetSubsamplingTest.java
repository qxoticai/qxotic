package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The tiled subsampling front end against itself untiled: each tile recomputes its receptive
 * field's halo, and zero padding must come only from the real edges, never a tile boundary. Lengths
 * land on and around tile boundaries, at both ends of a frame's mel range.
 */
class ParakeetSubsamplingTest {

    private static Arena arena;
    private static ParakeetEncoder encoder;

    @BeforeAll
    static void load() throws IOException {
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-f16.gguf");
        arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(channel, model.toString());
            encoder =
                    ParakeetEncoder.load(
                            gguf, ModelLoader.loadTensors(channel, gguf, arena), arena);
        }
    }

    @AfterAll
    static void unload() {
        if (arena != null) arena.close();
    }

    @ParameterizedTest(name = "{0} frames, mel slack {1}, tile {2}")
    @CsvSource({
        "1, 0, 1",
        "1, 7, 64",
        "63, 0, 64",
        "63, 7, 64",
        "64, 0, 64",
        "64, 7, 64",
        "65, 0, 64",
        "65, 7, 7",
        "128, 0, 64",
        "129, 0, 64",
        "129, 7, 1",
    })
    void tilesMatchTheWholeWindow(int frames, int slack, int tile) {
        // encoder frames = ceil(melFrames / 8): 8*frames - slack mel frames still gives frames
        int melFrames = 8 * frames - slack;
        float[] pcm = new float[(melFrames - 1) * encoder.config().hop()];
        Random random = new Random(frames * 31L + slack);
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) (0.1 * random.nextGaussian());
        assertEquals(frames, encoder.frames(pcm.length));
        assertEquals(melFrames, encoder.melFrames(pcm.length));
        int valid =
                ParakeetEncoder.subsampled(
                        ParakeetEncoder.subsampled(ParakeetEncoder.subsampled(melFrames - 1)));
        float[] mel = encoder.mel(pcm, 0, pcm.length, null, false);

        float[] tiled = preEncode(mel, melFrames, frames, valid, tile);
        float[] whole = preEncode(mel, melFrames, frames, valid, Integer.MAX_VALUE);
        double error = 0, norm = 0;
        for (int i = 0; i < whole.length; i++) {
            assertTrue(Float.isFinite(tiled[i]), "non-finite at " + i);
            error += Math.pow(tiled[i] - whole[i], 2);
            norm += Math.pow(whole[i], 2);
        }
        // a halo or edge mistake is an O(1) error on whole frames; float noise is ~1e-7
        assertTrue(
                Math.sqrt(error / Math.max(norm, 1e-30)) < 1e-6,
                "relative L2 " + Math.sqrt(error / norm));
    }

    private static float[] preEncode(float[] mel, int melFrames, int frames, int valid, int tile) {
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            var x = encoder.preEncode(mel, melFrames, frames, valid, tile, new Workspace(scratch));
            return Views.toFloatArray(x, "pre-encode");
        } finally {
            Arenas.close(scratch);
        }
    }
}

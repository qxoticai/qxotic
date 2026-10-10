package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.media.Media;
import org.junit.jupiter.api.Test;

/**
 * Media preprocessing pinned to the reference libraries' outputs: Google's image sizing (JAX {@code
 * gemma4/vision/_preprocessing.py}) and Pillow's bicubic it resizes with, and miniaudio's
 * resampling to 16 kHz for audio at other rates (the reference takes 16 kHz; this is llama.cpp's
 * decoder).
 */
class MediaPreprocessTest {
    /** s16 samples, as a WAV holds them. */
    private static final short[] SIGNAL = {
        -10005, -2086, 5833, -6259, 1660, 9579, -2513, 5406, -6686, 1233, 9152, -2940, 4979, -7113,
        806, 8725, -3367, 4552, -7540, 379, 8298, -3794, 4125, -7967
    };

    @Test
    void downsamplesLikeMiniaudio() {
        assertArrayEquals(
                new float[] {
                    0.0f,
                    -0.13980555534362793f,
                    -0.10239523649215698f,
                    0.04993904381990433f,
                    -0.049288779497146606f,
                    0.17982637882232666f,
                    -0.0999956801533699f,
                    0.0029415488243103027f,
                    0.15414196252822876f,
                    0.0160975381731987f,
                    -0.07273949682712555f,
                    0.09029562026262283f,
                    0.08547673374414444f,
                    -0.0849025696516037f,
                    0.010626953095197678f,
                    0.10734517127275467f
                },
                AudioPreprocess.toMono16k(audio(SIGNAL.length, 24_000)));
    }

    @Test
    void upsamplesLikeMiniaudio() {
        assertArrayEquals(
                new float[] {
                    0.0f,
                    -0.01434751134365797f,
                    -0.08608506619930267f,
                    -0.21123075485229492f,
                    -0.26305878162384033f,
                    -0.14880119264125824f,
                    0.033650174736976624f,
                    0.13332977890968323f,
                    0.08378425985574722f,
                    -0.07194794714450836f,
                    -0.156357079744339f,
                    -0.0442119762301445f,
                    0.15068989992141724f,
                    0.25193652510643005f,
                    0.19691523909568787f,
                    0.04045873135328293f,
                    -0.04150893911719322f,
                    0.042264051735401154f,
                    0.09259375929832458f,
                    -0.0506252646446228f,
                    -0.17890571057796478f,
                    -0.07256139814853668f
                },
                AudioPreprocess.toMono16k(audio(11, 8_000)));
    }

    @Test
    void logMelAddsTheFloorLikeTheReference() {
        // a quiet 440 Hz tone keeps every band near the 0.001 floor, where log(mel + floor)
        // differs from a clamp, log(max(mel, floor)) = -6.9078 (llama.cpp's form)
        float[] pcm = new float[1600];
        for (int i = 0; i < pcm.length; i++)
            pcm[i] = (float) (0.0005 * Math.sin(2 * Math.PI * 440 * i / 16_000));
        AudioPreprocess.MelChunk mel = new AudioPreprocess(128).logMel(pcm).getFirst();
        assertEquals(9, mel.frames());
        float[][] expected = { // frame, band, HF Gemma4AudioFeatureExtractor's value
            {1, 3, -6.888944149017334f},
            {4, 12, -6.892884254455566f},
            {5, 20, -6.369950294494629f},
            {8, 20, -6.368711471557617f},
            {8, 64, -6.906982898712158f}
        };
        for (float[] e : expected)
            assertEquals(
                    e[2],
                    mel.data()[(int) e[0] * 128 + (int) e[1]],
                    1e-4f,
                    "mel " + e[0] + "," + e[1]);
    }

    @Test
    void sizesLikeTheReference() {
        int[][] cases = { // width, height, budget -> width, height (get_target_dimensions)
            {640, 480, 280, 912, 672},
            {519, 600, 280, 720, 816},
            {60, 40, 280, 960, 624}, // small images grow to fill the budget
            {4000, 3000, 280, 912, 672},
            {2000, 30, 280, 6528, 96},
            {30, 2000, 280, 96, 6528},
            {640, 480, 70, 432, 336},
            {640, 480, 1120, 1824, 1344}
        };
        for (int[] c : cases)
            assertArrayEquals(
                    new int[] {c[3], c[4]},
                    VisionPreprocess.targetSize(c[0], c[1], 48, c[2]),
                    c[0] + "x" + c[1] + " budget " + c[2]);
    }

    @Test
    void anUnchangedSizeIsAnExactCopy() {
        float[] rgb = new float[5 * 4 * 3];
        for (int i = 0; i < rgb.length; i++) rgb[i] = ((i * 37) % 256) / 255f;
        float[] planes = VisionPreprocess.toCHW(new Media.Image(rgb, 4, 5, 3), 5, 4);
        for (int i = 0; i < 5 * 4; i++)
            for (int c = 0; c < 3; c++)
                assertEquals(rgb[i * 3 + c], planes[c * 20 + i], "pixel " + i);
    }

    @Test
    void resizesWithPillowsBicubic() {
        int width = 9, height = 7;
        float[] rgb = new float[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                    rgb[(y * width + x) * 3 + c] = ((x * 37 + y * 91 + c * 53) % 256) / 255f;
        float[] planes = VisionPreprocess.toCHW(new Media.Image(rgb, height, width, 3), 6, 5);
        int[] expected = { // Pillow's Image.resize((6, 5), BICUBIC), as planes
            32, 94, 161, 159, 115, 42, 166, 110, 119, 96, 113, 177, 66, 94, 120, 157, 152, 74, 168,
            145, 104, 55, 128, 181, 102, 91, 122, 180, 72, 114, 90, 156, 182, 162, 59, 111, 159,
            112, 136, 105, 183, 131, 105, 129, 149, 200, 94, 110, 177, 142, 67, 119, 187, 159, 120,
            112, 194, 119, 113, 111, 157, 146, 179, 106, 101, 171, 119, 90, 104, 176, 180, 125, 136,
            169, 146, 73, 132, 150, 161, 106, 111, 171, 122, 172, 115, 185, 181, 99, 86, 133
        };
        float[] values = new float[expected.length];
        for (int i = 0; i < values.length; i++) values[i] = expected[i] / 255.0f;
        assertArrayEquals(values, planes);
    }

    private static Media.Audio audio(int samples, int sampleRate) {
        float[] pcm = new float[samples];
        for (int i = 0; i < samples; i++) pcm[i] = SIGNAL[i] / 32768f;
        return new Media.Audio(pcm, sampleRate, 1);
    }
}

package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The media-positions law: {@code positions(media)} - the preprocessing PLAN's count, computed
 * without any tower run - must equal the rows {@code encode} actually produces, for every image
 * shape (smart-resize tiers) and audio duration/rate (mono-16k frame arithmetic). This is the drift
 * guard that keeps token estimates honest against the encoders.
 */
@Tag("integration")
class MediaPositionsIT {

    @Test
    void imagePlanMatchesEncodedRows() throws Exception {
        var mmproj = com.qxotic.jinfer.testkit.ModelFixture.GEMMA4_E2B_MMPROJ.require();
        Gemma4Vision vision = Gemma4Vision.loadModel(mmproj, Arena.ofAuto());
        int[][] shapes = {{256, 256}, {640, 480}, {111, 333}, {1600, 900}};
        for (int[] wh : shapes) {
            Media.Image img = solidImage(wh[0], wh[1]);
            int plan = vision.positions(img);
            FloatTensor rows = vision.encode(img);
            assertEquals(
                    plan,
                    (int) (rows.size() / vision.modelDim),
                    wh[0] + "x" + wh[1] + ": plan vs encoded rows");
        }
    }

    @Test
    void audioPlanMatchesEncodedRows() {
        var mmproj = com.qxotic.jinfer.testkit.ModelFixture.GEMMA4_12B_MMPROJ.require();
        Gemma4Audio audio;
        try {
            audio = Gemma4Audio.loadModel(mmproj, Arena.ofAuto());
        } catch (Exception e) {
            Assumptions.abort("mmproj has no audio projection: " + e);
            return;
        }
        double[][] clips = {{1.3, 16000}, {0.7, 44100}, {2.0, 8000}};
        for (double[] c : clips) {
            Media.Audio clip = sine(440, c[0], (int) c[1]);
            int plan = audio.positions(clip);
            FloatTensor rows = audio.encode(clip);
            assertEquals(
                    plan,
                    (int) (rows.size() / audio.modelDim),
                    c[0] + "s @" + (int) c[1] + "Hz: plan vs encoded rows");
        }
    }

    private static Media.Image solidImage(int w, int h) {
        float[] v = new float[h * w * 3];
        for (int i = 0; i < h * w; i++) {
            v[i * 3] = 0.8f;
            v[i * 3 + 2] = 0.2f;
        }
        return new Media.Image(v, h, w, 3);
    }

    private static Media.Audio sine(double hz, double seconds, int rate) {
        int n = (int) (rate * seconds);
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) pcm[i] = (float) (0.4 * Math.sin(2 * Math.PI * hz * i / rate));
        return new Media.Audio(pcm, rate, 1);
    }
}

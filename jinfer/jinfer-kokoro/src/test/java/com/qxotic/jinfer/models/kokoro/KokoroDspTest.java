package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

final class KokoroDspTest {

    private static final float[] WEIGHT = {0.2f, -0.1f, 0.3f, 0.1f, -0.2f, 0.4f, 0.2f, -0.3f, 0.1f};

    @Test
    void sourceStftHasExpectedDimensionsAndFiniteValues() {
        KokoroDsp.Spectrum spectrum =
                KokoroDsp.sourceStft(new float[] {120, 140}, WEIGHT, 0.01f, new Random(7), null);

        assertEquals(11, spectrum.magnitude().length);
        assertEquals(121, spectrum.magnitude()[0].length);
        assertEquals(11, spectrum.phase().length);
        assertEquals(121, spectrum.phase()[0].length);
        for (float[] channels :
                new float[][] {flatten(spectrum.magnitude()), flatten(spectrum.phase())})
            for (float value : channels) assertTrue(Float.isFinite(value));
    }

    @Test
    void sourceStftUsesJavaRandomSeedDeterministically() {
        KokoroDsp.Spectrum first =
                KokoroDsp.sourceStft(new float[] {0, 100}, WEIGHT, 0, new Random(1234), null);
        KokoroDsp.Spectrum second =
                KokoroDsp.sourceStft(new float[] {0, 100}, WEIGHT, 0, new Random(1234), null);

        assertArrayEquals(flatten(first.magnitude()), flatten(second.magnitude()));
        assertArrayEquals(flatten(first.phase()), flatten(second.phase()));
    }

    @Test
    void zeroIstftHasGeneratorLengthAndFiniteOutput() {
        float[] output = KokoroDsp.istft(new float[11][4], new float[11][4], null);

        assertEquals(15, output.length);
        assertArrayEquals(new float[15], output);
        for (float value : output) assertTrue(Float.isFinite(value));
    }

    @Test
    void istftIgnoresDcPhase() {
        float[][] magnitude = new float[11][4];
        float[][] zeroPhase = new float[11][4];
        float[][] changedDcPhase = new float[11][4];
        for (int frame = 0; frame < 4; frame++) {
            magnitude[0][frame] = 1;
            changedDcPhase[0][frame] = (frame + 1) * 0.7f;
        }

        float[] expected = KokoroDsp.istft(magnitude, zeroPhase, null);
        float[] actual = KokoroDsp.istft(magnitude, changedDcPhase, null);

        assertArrayEquals(expected, actual);
        assertTrue(actual[0] > 0);
    }

    @Test
    void oneFrameIstftIsEmpty() {
        assertEquals(0, KokoroDsp.istft(new float[11][1], new float[11][1], null).length);
    }

    private static float[] flatten(float[][] values) {
        int columns = values[0].length;
        float[] flattened = new float[values.length * columns];
        for (int row = 0; row < values.length; row++)
            System.arraycopy(values[row], 0, flattened, row * columns, columns);
        return flattened;
    }
}

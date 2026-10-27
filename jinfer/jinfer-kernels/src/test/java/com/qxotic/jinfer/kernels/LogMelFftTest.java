package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The FFT under LogMel against a direct DFT in double: radix-2 sizes and odd leaves alike. */
class LogMelFftTest {

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 16, 25, 400, 512})
    void matchesADirectDft(int n) {
        LogMel mel =
                new LogMel(
                        new LogMel.Spec(
                                n, 1, 1, new float[n], new float[n / 2 + 1], 0f, 2f, 0f, 0f));
        float[] input = new float[n * 2], output = new float[n * 8];
        Random random = new Random(n);
        for (int i = 0; i < n; i++) input[i] = random.nextFloat() * 2 - 1;
        float[] signal = input.clone();
        mel.fftReal(input, 0, n, output, 0);
        for (int k = 0; k <= n / 2; k++) {
            double re = 0, im = 0;
            for (int j = 0; j < n; j++) {
                re += signal[j] * Math.cos(2 * Math.PI * k * j / n);
                im -= signal[j] * Math.sin(2 * Math.PI * k * j / n);
            }
            assertEquals(re, output[2 * k], 1e-4, "re " + k);
            assertEquals(im, output[2 * k + 1], 1e-4, "im " + k);
        }
    }
}

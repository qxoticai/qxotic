package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import org.junit.jupiter.api.Test;

/**
 * {@link Resample} on its own: the rules every vision tower that resizes relies on - 8-bit RGB out,
 * alpha dropped and grey replicated, an unchanged size an exact copy, both weight roundings, and a
 * target of zero refused. Pixel-exact agreement with Pillow/torchvision is pinned by the model
 * parity suites; these are the contract's edges.
 */
final class ResampleTest {

    private static Media.Image rgb(int width, int height, int[] pixels) {
        float[] values = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) values[i] = pixels[i] / 255f;
        return new Media.Image(values, height, width, 3);
    }

    /** An unchanged size is the image's own 8-bit RGB, byte for byte, no filter applied. */
    @Test
    void anUnchangedSizeIsTheExactImage() {
        int[] pixels = {0, 10, 20, 30, 40, 50, 255, 128, 1, 2, 3, 4};
        Media.Image image = rgb(2, 2, pixels);
        assertArrayEquals(
                pixels, Resample.bicubic(image, 2, 2, false), "Pillow weights, no resize");
        assertArrayEquals(
                pixels, Resample.bicubic(image, 2, 2, true), "torchvision weights, no resize");
    }

    /** An RGBA image drops alpha and replicates grey, as stb_image loads it. */
    @Test
    void alphaIsDroppedAndGreyReplicated() {
        // one pixel, RGBA: grey 77, alpha 200 -> RGB 77,77,77
        float g = 77f / 255f;
        Media.Image rgba = new Media.Image(new float[] {g, g, g, 200f / 255f}, 1, 1, 4);
        assertArrayEquals(new int[] {77, 77, 77}, Resample.bicubic(rgba, 1, 1, false));

        // a one-channel image is grey at every lane
        Media.Image grey = new Media.Image(new float[] {g}, 1, 1, 1);
        assertArrayEquals(new int[] {77, 77, 77}, Resample.bicubic(grey, 1, 1, false));
    }

    /** A solid colour resizes to the same colour: the weights sum to one, either rounding. */
    @Test
    void aSolidColourSurvivesBothRoundings() {
        int[] pixels = new int[8 * 8 * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = 137;
        Media.Image image = rgb(8, 8, pixels);
        for (boolean torch : new boolean[] {false, true}) {
            for (int size : new int[] {1, 3, 16, 37}) {
                int[] out = Resample.bicubic(image, size, size, torch);
                assertEquals(size * size * 3, out.length, "size " + size);
                for (int v : out)
                    assertEquals(137, v, 1, "solid at " + size + "x" + size + ", torch=" + torch);
            }
        }
    }

    /** Upscaling a one-pixel and a two-pixel pattern keeps the endpoints and stays in 0..255. */
    @Test
    void everyOutputIsInRange() {
        int[] pixels = new int[3 * 2 * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (i * 37) % 256;
        Media.Image image = rgb(3, 2, pixels);
        for (boolean torch : new boolean[] {false, true}) {
            for (int[] target : new int[][] {{1, 1}, {2, 3}, {7, 5}, {11, 9}, {40, 1}, {1, 40}}) {
                int[] out = Resample.bicubic(image, target[0], target[1], torch);
                assertEquals(target[0] * target[1] * 3, out.length);
                for (int v : out) assertTrue(v >= 0 && v <= 255, "out of range: " + v);
            }
        }
    }

    /** The two roundings differ only in the last bit on a non-trivial image, never widely. */
    @Test
    void theTwoRoundingsStayClose() {
        int w = 9, h = 7;
        int[] pixels = new int[w * h * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (i * 53 + 17) % 256;
        Media.Image image = rgb(w, h, pixels);
        int[] pillow = Resample.bicubic(image, 5, 4, false);
        int[] torch = Resample.bicubic(image, 5, 4, true);
        assertEquals(pillow.length, torch.length);
        for (int i = 0; i < pillow.length; i++)
            assertTrue(
                    Math.abs(pillow[i] - torch[i]) <= 1,
                    "index " + i + ": pillow " + pillow[i] + ", torch " + torch[i]);
    }

    @Test
    void anEmptyTargetIsRefused() {
        Media.Image image = rgb(2, 2, new int[12]);
        assertThrows(IllegalArgumentException.class, () -> Resample.bicubic(image, 0, 4, false));
        assertThrows(IllegalArgumentException.class, () -> Resample.bicubic(image, 4, 0, false));
        assertThrows(IllegalArgumentException.class, () -> Resample.bicubic(image, -1, 4, false));
    }
}
